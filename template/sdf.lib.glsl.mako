float sphere(vec3 v, float r) {
	return length(v) - r;
}

float torus(vec3 v, vec2 t) {
	vec2 q = v2(length(v.xz)-t.x,v.y);
	return length(q)-t.y;
}

float box(vec3 v, vec3 b) {
  vec3 q = fabs(v) - b;
  return length(max(q,0.0)) + min(max(q.x,max(q.y,q.z)),0.0);
}

vec3 flip_xy(vec3 v) {
	return v3(v.y, v.x, v.z);
}

vec3 flip_yz(vec3 v) {
	return v3(v.x, v.z, v.y);
}

vec3 rotate_x(vec3 v, float r) {
	v.yz = cos(r)*v.yz + sin(r)*v2(v.z, -v.y);
	return v;
}

vec3 rotate_y(vec3 v, float r) {
	v.xz = cos(r)*v.xz + sin(r)*v2(v.z, -v.x);
	return v;
}

vec3 rotate_z(vec3 v, float r) {
	v.xy = cos(r)*v.xy + sin(r)*v2(v.y, -v.x);
	return v;
}

vec3 translate(vec3 v, vec3 w) {
	return v - w;
}

float rounded(float a, float r) {
	return a - r;
}

float sunify(float a, float b, float k) {
	float h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0);
	return mix(b, a, h) - k * h * (1 - h);
}

float ssub(float b, float a, float k) {
	float h = clamp(0.5 - 0.5*(b+a)/k, 0.0, 1.0);
	return mix(b, -a, h) + k*h*(1.0-h);
}

float sintersect(float a, float b, float k) {
	float h = clamp(0.5 - 0.5*(b-a)/k, 0.0, 1.0);
	return mix(b, a, h) + k*h*(1.0-h);
}

float sub(float a, float b) {
	return max(-b, a);
}

float unify(float a, float b) {
	return min(a, b);
}

float intersect(float a, float b) {
	return max(a, b);
}
