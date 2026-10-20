/**
 * Custom sigma.js v3 WebGL node program for rendering diamond shapes.
 * Used for ADR nodes in the Drug–ADR Network graph.
 *
 * Based on sigma v3's NodePointProgram — same vertex shader, custom fragment
 * shader that clips to a diamond (|x| + |y| <= 0.5).
 */
import { NodeProgram } from 'sigma/rendering';
import { floatColor } from 'sigma/utils';

// language=GLSL
const VERTEX_SHADER_SOURCE = /*glsl*/ `
attribute vec4 a_id;
attribute vec4 a_color;
attribute vec2 a_position;
attribute float a_size;

uniform float u_sizeRatio;
uniform float u_pixelRatio;
uniform mat3 u_matrix;

varying vec4 v_color;
varying float v_border;

const float bias = 255.0 / 254.0;

void main() {
  gl_Position = vec4(
    (u_matrix * vec3(a_position, 1)).xy,
    0,
    1
  );

  // Multiply the point size twice:
  //  - x SCALING_RATIO to correct the canvas scaling
  //  - x 2 to correct the formulae
  gl_PointSize = a_size / u_sizeRatio * u_pixelRatio * 2.0;

  v_border = (0.5 / a_size) * u_sizeRatio;

  #ifdef PICKING_MODE
  v_color = a_id;
  #else
  v_color = a_color;
  #endif

  v_color.a *= bias;
}
`;

// language=GLSL — diamond shape via Manhattan distance
const FRAGMENT_SHADER_SOURCE = /*glsl*/ `
precision mediump float;

varying vec4 v_color;
varying float v_border;

const vec4 transparent = vec4(0.0, 0.0, 0.0, 0.0);

void main(void) {
  vec2 m = gl_PointCoord - vec2(0.5, 0.5);
  // Diamond: |x| + |y| <= 0.5
  float dist = 0.5 - (abs(m.x) + abs(m.y));

  #ifdef PICKING_MODE
  if (dist > v_border)
    gl_FragColor = v_color;
  else
    gl_FragColor = transparent;
  #else
  float t = 0.0;
  if (dist > v_border)
    t = 1.0;
  else if (dist > 0.0)
    t = dist / v_border;

  gl_FragColor = mix(transparent, v_color, t);
  #endif
}
`;

const { UNSIGNED_BYTE, FLOAT } = WebGLRenderingContext;
const UNIFORMS = ['u_sizeRatio', 'u_pixelRatio', 'u_matrix'];

export default class NodeDiamondProgram extends NodeProgram {
  getDefinition() {
    return {
      VERTICES: 1,
      VERTEX_SHADER_SOURCE,
      FRAGMENT_SHADER_SOURCE,
      METHOD: WebGLRenderingContext.POINTS,
      UNIFORMS,
      ATTRIBUTES: [
        { name: 'a_position', size: 2, type: FLOAT },
        { name: 'a_size', size: 1, type: FLOAT },
        { name: 'a_color', size: 4, type: UNSIGNED_BYTE, normalized: true },
        { name: 'a_id', size: 4, type: UNSIGNED_BYTE, normalized: true },
      ],
    };
  }

  processVisibleItem(nodeIndex, startIndex, data) {
    const array = this.array;
    array[startIndex++] = data.x;
    array[startIndex++] = data.y;
    array[startIndex++] = data.size;
    array[startIndex++] = floatColor(data.color);
    array[startIndex++] = nodeIndex;
  }

  setUniforms({ sizeRatio, pixelRatio, matrix }, { gl, uniformLocations }) {
    const { u_sizeRatio, u_pixelRatio, u_matrix } = uniformLocations;
    gl.uniform1f(u_pixelRatio, pixelRatio);
    gl.uniform1f(u_sizeRatio, sizeRatio);
    gl.uniformMatrix3fv(u_matrix, false, matrix);
  }
}
