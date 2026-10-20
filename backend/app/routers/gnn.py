"""
GNN Router — training status, trigger training, get predictions.
Uses SYNC database session for fast JOIN queries.
"""
from fastapi import APIRouter, Query, Depends
from typing import Optional
from sqlalchemy import text
from sqlalchemy.orm import Session
from app.database import get_sync_db
from app.routers.utils import build_quarter_filter

router = APIRouter()


@router.get("/status")
async def gnn_status():
    """Get GNN training status for all age groups."""
    from app.gnn.trainer import training_status
    return training_status


@router.post("/train/{age_group}")
def train_gnn(
    age_group: str,
    quarter: str = Query("ALL"),
    epochs: int = Query(100, ge=10, le=500),
    db: Session = Depends(get_sync_db),
):
    """Trigger GNN training for a specific age group."""
    import pandas as pd
    from app.gnn.trainer import train_gnn_for_age_group, training_status

    key = f"{age_group}_{quarter}"
    if key in training_status and training_status[key].get("status") == "training":
        return {"status": "already_training", "details": training_status[key]}

    q_filter, q_params = build_quarter_filter(quarter)

    demo_result = db.execute(text(
        f"SELECT primaryid, age_group FROM demographics "
        f"WHERE age_group IS NOT NULL AND is_superseded = FALSE {q_filter}"
    ), q_params)
    demo_df = pd.DataFrame(demo_result.fetchall(), columns=["primaryid", "age_group"])

    drug_result = db.execute(text(
        f"SELECT primaryid, drugname_normalized, role_cod FROM drugs "
        f"WHERE is_superseded = FALSE {q_filter}"
    ), q_params)
    drug_df = pd.DataFrame(drug_result.fetchall(), columns=["primaryid", "drugname_normalized", "role_cod"])

    reac_result = db.execute(text(
        f"SELECT primaryid, pt_term FROM reactions WHERE is_superseded = FALSE {q_filter}"
    ), q_params)
    reac_df = pd.DataFrame(reac_result.fetchall(), columns=["primaryid", "pt_term"])

    if demo_df.empty or drug_df.empty or reac_df.empty:
        return {"status": "error", "message": "No data available for training"}

    result = train_gnn_for_age_group(
        demo_df, drug_df, reac_df,
        model_key=key,  # Pass the quarter-aware key so it trains isolated models!
        epochs=epochs,
    )
    result["quarter"] = quarter
    result["age_group"] = age_group

    if quarter != "ALL" and result.get("status") == "completed":
        try:
            db.execute(text("UPDATE loaded_quarters SET gnn_trained = TRUE WHERE quarter = :q"), {"q": quarter})
            db.commit()
        except Exception as e:
            db.rollback()
            print(f"Failed to update gnn_trained flag: {e}")

    return result


@router.get("/predictions/{age_group}")
async def get_predictions(
    age_group: str,
    quarter: str = Query("ALL"),
    top_k: int = Query(50, le=200),
):
    """Get top-K novel drug-ADR predictions for an age group."""
    from app.gnn.predictor import predict_novel_links
    
    key = f"{age_group}_{quarter}"
    predictions = predict_novel_links(key, top_k=top_k)
    if not predictions:
        return {
            "status": "no_model",
            "message": f"No trained model for {age_group} in {quarter}. Train first via POST /api/gnn/train/{age_group}?quarter={quarter}",
            "predictions": [],
        }

    return {
        "status": "ok",
        "age_group": age_group,
        "quarter": quarter,
        "count": len(predictions),
        "predictions": predictions,
    }


@router.get("/graph-data/{age_group}")
def get_graph_data(
    age_group: str,
    quarter: str = Query("ALL"),
    db: Session = Depends(get_sync_db),
):
    """Get graph data for Cytoscape visualization."""
    q_filter_dr, q_params = build_quarter_filter(quarter, "dr.")
    q_params["age_group"] = age_group
    q_params["q_filter_sc"] = quarter
    # For signal cache, if we only compute signals globally, we just join. But if quarter string is provided, sc.quarter_filter handles it.
    
    query = f"""
        SELECT
            drugname_normalized as drug,
            pt_term as adr,
            n11 as count,
            COALESCE(prr, 0) as prr,
            COALESCE(ror, 0) as ror,
            CASE WHEN id IS NOT NULL AND (
                (prr >= 2 AND prr_chi2 >= 4) OR
                (ror >= 2 AND ror_ci_lower > 1) OR
                (ic025 > 0) OR
                (eb05 > 2)
            ) THEN true ELSE false END as is_signal
        FROM signal_cache
        WHERE age_group = :age_group
        AND quarter_filter = :q_filter_sc
        ORDER BY count DESC
        LIMIT 500
    """

    result = db.execute(text(query), q_params)
    rows = result.fetchall()

    nodes = {}
    edges = []

    for row in rows:
        drug, adr, count, prr, ror, is_signal = row

        if drug not in nodes:
            nodes[drug] = {"id": drug, "type": "drug", "count": 0}
        nodes[drug]["count"] += count

        if adr not in nodes:
            nodes[adr] = {"id": adr, "type": "adr", "count": 0}
        nodes[adr]["count"] += count

        edges.append({
            "source": drug,
            "target": adr,
            "count": count,
            "prr": float(prr) if prr else 0,
            "ror": float(ror) if ror else 0,
            "is_signal": is_signal,
        })

    # ── Fetch role_cod breakdown (PS/SS/C/I) for each drug node ──
    drug_names = [n for n, v in nodes.items() if v["type"] == "drug"]
    role_map = {}  # drug_name -> {"PS": count, "SS": count, ...}

    if drug_names:
        q_filter_d, q_params_d = build_quarter_filter(quarter, "d.")
        role_query = f"""
            SELECT d.drugname_normalized, d.role_cod, COUNT(*) as cnt
            FROM drugs d
            JOIN demographics dem ON d.primaryid = dem.primaryid
                AND dem.is_superseded = FALSE
            WHERE d.is_superseded = FALSE
            AND dem.age_group = :age_group
            AND d.drugname_normalized = ANY(:drug_list)
            AND d.role_cod IN ('PS', 'SS', 'C', 'I')
            {q_filter_d}
            GROUP BY d.drugname_normalized, d.role_cod
            ORDER BY d.drugname_normalized, cnt DESC
        """
        q_params_d["age_group"] = age_group
        q_params_d["drug_list"] = drug_names

        try:
            role_result = db.execute(text(role_query), q_params_d)
            for drug_name, role, cnt in role_result.fetchall():
                if drug_name not in role_map:
                    role_map[drug_name] = {}
                role_map[drug_name][role] = cnt
        except Exception as e:
            print(f"[GNN] role_cod query failed (non-fatal): {e}")

    # Attach role_counts to drug nodes
    for name, node in nodes.items():
        if node["type"] == "drug":
            node["role_counts"] = role_map.get(name, None)

    return {
        "nodes": list(nodes.values()),
        "edges": edges,
        "age_group": age_group,
    }

