#extension GL_EXT_control_flow_attributes : require

bool isLightSource(vec3 emission){
	if(length(emission)>0.01) return true;
	else return false;
}

vec3 getSampledReflectedDirection(vec3 cameraPos,vec3 inRay,vec3 normal,vec2 uv,float seed){
    inRay=inRay-cameraPos;
    vec3 Ray=reflect(inRay,normal);
    float theta=acos(1-random(uv,seed));  //[0,pi/2]
    float phi=2*M_PI*random(vec2(uv.y,uv.x),seed);
    vec3 RandomRay=vec3(sin(theta)*cos(phi),sin(theta)*sin(phi),cos(theta));
    float weight=0.5;  //reflection rate
    return normalize(weight*Ray+(1-weight)*normalize(RandomRay));
}