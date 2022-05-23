#version 460
#extension GL_ARB_separate_shader_objects : enable
#extension GL_EXT_ray_tracing : enable
#extension GL_EXT_ray_query : enable
#extension GL_EXT_debug_printf : enable
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_nonuniform_qualifier: require

#include "../includes/types.glsl"
#include "../includes/random.glsl"
#include "../includes/utils.glsl"

layout(binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
    QuadArealignt qLight;
    vec4 cameraPos;
    uint frameCount;
    uint mode;  //denoising algorithm   1:raw  2:mvec 3:svgf 4:ours 5: ground truth
} ubo;

layout(binding = 1) uniform sampler2D texSampler;
layout (binding = 2) uniform sampler2D textures[];
layout(binding = 3) buffer MaterialIndexBuffer { Primitive data[]; } materialIndexBuffer;
layout(binding = 4) buffer MaterialBuffer { Material data[]; } materialBuffer;
layout(binding = 5,set=0) uniform accelerationStructureEXT topLevelAS;
layout(binding = 6) buffer VertexBuffer { Vertex data[]; } vertexBuffer;
layout(binding = 7) buffer IndexBuffer { uint data[]; } indexBuffer;
layout (binding = 8, rgba32f) uniform image2D historyColorImages[];  //0:directAlbedo  1:directIR 2:indirectAlbedo 3:indirectIR 4:normal 5:world 6:imageVar
layout (binding = 9, r32f) uniform image2D historyDepthImage;

layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec3 fragNormal;
layout(location = 2) in vec3 interpolatedPosition;
layout(location = 3) in vec2 fragTexCoord;

layout(location = 0) out vec4 outColor;
layout(location = 1) out vec4 outDirectIr;
layout(location = 2) out vec4 outIndAlbedo;
layout(location = 3) out vec4 outIndIr;
layout(location = 4) out vec4 outNormal;
layout(location = 5) out vec4 outWorldPos;
layout(location = 6) out vec4 outDepth;

vec3 fragPos;
bool isShadow=false;
vec2 RayHitPointFragCoord;

float weight(vec2 p,vec2 q);
float w_depth(vec2 p,vec2 q);
float w_normal(vec2 p,vec2 q);
float w_lumin(vec2 p,vec2 q);
vec4 variance(vec2 p);
vec4 aTrous_indirectIr(vec2 p);
vec4 aTrous_indirectAlbedo(vec2 p);
vec4 aTrous_directIr(vec2 p);

void main() {
    vec3 surfaceColor=vec3(0.0,0.0,0.0);
    vec3 directIr = vec3(0.0, 0.0, 0.0);
    vec3 indirectAlbedo = vec3(0.0, 0.0, 0.0);
    vec3 indirectIr = vec3(0.0, 0.0, 0.0);

    surfaceColor=imageLoad(historyColorImages[0], ivec2(gl_FragCoord.xy)).xyz;
    indirectAlbedo= ubo.mode==4?aTrous_indirectAlbedo(gl_FragCoord.xy).xyz:imageLoad(historyColorImages[2], ivec2(gl_FragCoord.xy)).xyz;
    outIndAlbedo=vec4(indirectAlbedo,1.0f);

    if(ubo.mode==3 ||ubo.mode==4){
            directIr=aTrous_directIr(gl_FragCoord.xy).xyz;
            outDirectIr=vec4(directIr,1.0);
            indirectIr=0.6*aTrous_indirectIr(gl_FragCoord.xy).xyz;
            outIndIr=vec4(indirectIr,1.0);
    }
    else{
        directIr=imageLoad(historyColorImages[1], ivec2(gl_FragCoord.xy)).xyz;
        indirectIr=imageLoad(historyColorImages[3], ivec2(gl_FragCoord.xy)).xyz;
    }

    outNormal=vec4(surfaceColor,1.0);
    outColor=vec4(directIr*surfaceColor+0.2*indirectIr*indirectAlbedo+surfaceColor*0.05,1.0);
    //outDirectIr=vec4(0.5,0.0,0.0,1.0);

    //if(isLightSource(materialBuffer.data[material_id].emission)) outColor = vec4(materialBuffer.data[material_id].emission,1.0f);
}


float w_depth(vec2 p,vec2 q){   //weight of depth in the edge stop function in SVGF
    float sigma_z=1.0;  //1.0 in the SVGF paper
    float epsil=0.01;
    //calculate gradient
    vec2 grad_p;
    float right=imageLoad(historyDepthImage,ivec2(p.x+1,p.y)).x;
    float left=imageLoad(historyDepthImage,ivec2(p.x-1,p.y)).x;
    float up=imageLoad(historyDepthImage,ivec2(p.x,p.y+1)).x;
    float down=imageLoad(historyDepthImage,ivec2(p.x,p.y-1)).x;
    grad_p.x=0.5*right-0.5*left;
    grad_p.y=0.5*up-0.5*down;
    //load p,q
    float depth_p=imageLoad(historyDepthImage,ivec2(p.x,p.y)).x;
    float depth_q=imageLoad(historyDepthImage,ivec2(q.x,q.y)).x;
    //caculate weight
    float weight=exp(-(abs(depth_p-depth_q)/sigma_z*dot(grad_p,(p-q))+epsil));
    return weight;
}

float w_normal(vec2 p,vec2 q){   //weight of normal in the edge stop function in SVGF
    float sigma_n=64;
    vec3 n_p=imageLoad(historyColorImages[4],ivec2(p.xy)).xyz;
    vec3 n_q=imageLoad(historyColorImages[4],ivec2(q.xy)).xyz;
    float weight=pow(max(0,dot(n_p,n_q)),sigma_n);
    return weight;
}

float w_lumin(vec2 p,vec2 q){//weight of Luminance in the edge stop function in SVGF
    float sigma_l=4;
    float epsil=0.01;
    float lumin_p=length(imageLoad(historyColorImages[1],ivec2(p.xy)).xyz);
    float lumin_q=length(imageLoad(historyColorImages[1],ivec2(q.xy)).xyz);
    float weight=exp(-abs(lumin_p-lumin_q)/(sigma_l*variance(p).z+epsil));
    return weight;
}

vec4 variance(vec2 p){
    float factor=0.001; //ËõÐ¡µ½0~1.0µÄ·¶Î§
    return imageLoad(historyColorImages[6],ivec2(p.xy))/factor;
}

float weight(vec2 p,vec2 q){
     return  w_normal(p,q)*w_lumin(p,q);//w_depth(p,q)*
}

vec4 aTrous_indirectIr(vec2 p){
    vec4 Numerator=vec4(0.0,0.0,0.0,1.0);
    vec4 Denominator=vec4(0.0,0.0,0.0,1.0);

    float level=8;
    vec4 Ir_00 = imageLoad(historyColorImages[3], ivec2(gl_FragCoord.x-level,gl_FragCoord.y-level));
    Numerator+=(1.0/16.0)*weight(gl_FragCoord.xy,vec2(gl_FragCoord.x-level,gl_FragCoord.y-level))*Ir_00;
    Denominator+=(1.0/16.0)*weight(gl_FragCoord.xy,vec2(gl_FragCoord.x-level,gl_FragCoord.y-level));

    vec4 Ir_01 = imageLoad(historyColorImages[3], ivec2(gl_FragCoord.x,gl_FragCoord.y-level));
    Numerator+=(1.0/8.0)*weight(gl_FragCoord.xy,vec2(gl_FragCoord.x,gl_FragCoord.y-level))*Ir_01;
    Denominator+=(1.0/8.0)*weight(gl_FragCoord.xy,vec2(gl_FragCoord.x,gl_FragCoord.y-level));

    vec4 Ir_02 = imageLoad(historyColorImages[3], ivec2(gl_FragCoord.x+level,gl_FragCoord.y-level));
    Numerator+=(1.0/16.0)*weight(gl_FragCoord.xy,vec2(gl_FragCoord.x+level,gl_FragCoord.y-level))*Ir_02;
    Denominator+=(1.0/16.0)*weight(gl_FragCoord.xy,vec2(gl_FragCoord.x+level,gl_FragCoord.y-level));

    vec4 Ir_10 = imageLoad(historyColorImages[3], ivec2(gl_FragCoord.x-level,gl_FragCoord.y));
    Numerator+=(1.0/8.0)*weight(gl_FragCoord.xy,vec2(gl_FragCoord.x-level,gl_FragCoord.y))*Ir_10;
    Denominator+=(1.0/8.0)*weight(gl_FragCoord.xy,vec2(gl_FragCoord.x-level,gl_FragCoord.y));

    vec4 Ir_11 = imageLoad(historyColorImages[3], ivec2(gl_FragCoord.xy));
    Numerator+=(1.0/4.0)*weight(gl_FragCoord.xy,vec2(gl_FragCoord.xy))*Ir_11;
    Denominator+=(1.0/4.0)*weight(gl_FragCoord.xy,vec2(gl_FragCoord.xy));

    vec4 Ir_12 = imageLoad(historyColorImages[3], ivec2(gl_FragCoord.x+level,gl_FragCoord.y));
    Numerator+=(1.0/8.0)*weight(gl_FragCoord.xy,vec2(gl_FragCoord.x+level,gl_FragCoord.y))*Ir_12;
    Denominator+=(1.0/8.0)*weight(gl_FragCoord.xy,vec2(gl_FragCoord.x+level,gl_FragCoord.y));

    vec4 Ir_20 = imageLoad(historyColorImages[3], ivec2(gl_FragCoord.x-level,gl_FragCoord.y+level));
    Numerator+=(1.0/16.0)*weight(gl_FragCoord.xy,vec2(gl_FragCoord.x-level,gl_FragCoord.y+level))*Ir_20;
    Denominator+=(1.0/16.0)*weight(gl_FragCoord.xy,vec2(gl_FragCoord.x-level,gl_FragCoord.y+level));

    vec4 Ir_21 = imageLoad(historyColorImages[3], ivec2(gl_FragCoord.x,gl_FragCoord.y+level));
    Numerator+=(1.0/8.0)*weight(gl_FragCoord.xy,vec2(gl_FragCoord.x,gl_FragCoord.y+level))*Ir_21;
    Denominator+=(1.0/8.0)*weight(gl_FragCoord.xy,vec2(gl_FragCoord.x,gl_FragCoord.y+level));

    vec4 Ir_22 = imageLoad(historyColorImages[3], ivec2(gl_FragCoord.x+level,gl_FragCoord.y+level));
    Numerator+=(1.0/16.0)*weight(gl_FragCoord.xy,vec2(gl_FragCoord.x+level,gl_FragCoord.y+level))*Ir_22;
    Denominator+=(1.0/16.0)*weight(gl_FragCoord.xy,vec2(gl_FragCoord.x+level,gl_FragCoord.y+level));

    //vec4 Ir=(1/4.0)*Ir_11+(1/8.0)*(Ir_01+Ir_10+Ir_12+Ir_21)+(1/16.0)*(Ir_00+Ir_02+Ir_20+Ir_22);

    return Numerator/Denominator;
}

vec4 aTrous_indirectAlbedo(vec2 p){
    vec4 Numerator=vec4(0.0,0.0,0.0,1.0);
    vec4 Denominator=vec4(0.0,0.0,0.0,1.0);

    float level=8;
    vec4 Ir_00 = imageLoad(historyColorImages[2], ivec2(gl_FragCoord.x-level,gl_FragCoord.y-level));
    Numerator+=(1.0/16.0)*weight(gl_FragCoord.xy,vec2(gl_FragCoord.x-level,gl_FragCoord.y-level))*Ir_00;
    Denominator+=(1.0/16.0)*weight(gl_FragCoord.xy,vec2(gl_FragCoord.x-level,gl_FragCoord.y-level));

    vec4 Ir_01 = imageLoad(historyColorImages[2], ivec2(gl_FragCoord.x,gl_FragCoord.y-level));
    Numerator+=(1.0/8.0)*weight(gl_FragCoord.xy,vec2(gl_FragCoord.x,gl_FragCoord.y-level))*Ir_01;
    Denominator+=(1.0/8.0)*weight(gl_FragCoord.xy,vec2(gl_FragCoord.x,gl_FragCoord.y-level));

    vec4 Ir_02 = imageLoad(historyColorImages[2], ivec2(gl_FragCoord.x+level,gl_FragCoord.y-level));
    Numerator+=(1.0/16.0)*weight(gl_FragCoord.xy,vec2(gl_FragCoord.x+level,gl_FragCoord.y-level))*Ir_02;
    Denominator+=(1.0/16.0)*weight(gl_FragCoord.xy,vec2(gl_FragCoord.x+level,gl_FragCoord.y-level));

    vec4 Ir_10 = imageLoad(historyColorImages[2], ivec2(gl_FragCoord.x-level,gl_FragCoord.y));
    Numerator+=(1.0/8.0)*weight(gl_FragCoord.xy,vec2(gl_FragCoord.x-level,gl_FragCoord.y))*Ir_10;
    Denominator+=(1.0/8.0)*weight(gl_FragCoord.xy,vec2(gl_FragCoord.x-level,gl_FragCoord.y));

    vec4 Ir_11 = imageLoad(historyColorImages[2], ivec2(gl_FragCoord.xy));
    Numerator+=(1.0/4.0)*weight(gl_FragCoord.xy,vec2(gl_FragCoord.xy))*Ir_11;
    Denominator+=(1.0/4.0)*weight(gl_FragCoord.xy,vec2(gl_FragCoord.xy));

    vec4 Ir_12 = imageLoad(historyColorImages[2], ivec2(gl_FragCoord.x+level,gl_FragCoord.y));
    Numerator+=(1.0/8.0)*weight(gl_FragCoord.xy,vec2(gl_FragCoord.x+level,gl_FragCoord.y))*Ir_12;
    Denominator+=(1.0/8.0)*weight(gl_FragCoord.xy,vec2(gl_FragCoord.x+level,gl_FragCoord.y));

    vec4 Ir_20 = imageLoad(historyColorImages[2], ivec2(gl_FragCoord.x-level,gl_FragCoord.y+level));
    Numerator+=(1.0/16.0)*weight(gl_FragCoord.xy,vec2(gl_FragCoord.x-level,gl_FragCoord.y+level))*Ir_20;
    Denominator+=(1.0/16.0)*weight(gl_FragCoord.xy,vec2(gl_FragCoord.x-level,gl_FragCoord.y+level));

    vec4 Ir_21 = imageLoad(historyColorImages[2], ivec2(gl_FragCoord.x,gl_FragCoord.y+level));
    Numerator+=(1.0/8.0)*weight(gl_FragCoord.xy,vec2(gl_FragCoord.x,gl_FragCoord.y+level))*Ir_21;
    Denominator+=(1.0/8.0)*weight(gl_FragCoord.xy,vec2(gl_FragCoord.x,gl_FragCoord.y+level));

    vec4 Ir_22 = imageLoad(historyColorImages[2], ivec2(gl_FragCoord.x+level,gl_FragCoord.y+level));
    Numerator+=(1.0/16.0)*weight(gl_FragCoord.xy,vec2(gl_FragCoord.x+level,gl_FragCoord.y+level))*Ir_22;
    Denominator+=(1.0/16.0)*weight(gl_FragCoord.xy,vec2(gl_FragCoord.x+level,gl_FragCoord.y+level));

    //vec4 Ir=(1/4.0)*Ir_11+(1/8.0)*(Ir_01+Ir_10+Ir_12+Ir_21)+(1/16.0)*(Ir_00+Ir_02+Ir_20+Ir_22);

    return Numerator/Denominator;
}

vec4 aTrous_directIr(vec2 p){
    vec4 Numerator=vec4(0.0,0.0,0.0,1.0);
    vec4 Denominator=vec4(0.0,0.0,0.0,1.0);

    float level=32;
    vec4 Ir_00 = imageLoad(historyColorImages[1], ivec2(gl_FragCoord.x-level,gl_FragCoord.y-level));
    Numerator+=(1.0/16.0)*weight(gl_FragCoord.xy,vec2(gl_FragCoord.x-level,gl_FragCoord.y-level))*Ir_00;
    Denominator+=(1.0/16.0)*weight(gl_FragCoord.xy,vec2(gl_FragCoord.x-level,gl_FragCoord.y-level));

    vec4 Ir_01 = imageLoad(historyColorImages[1], ivec2(gl_FragCoord.x,gl_FragCoord.y-level));
    Numerator+=(1.0/8.0)*weight(gl_FragCoord.xy,vec2(gl_FragCoord.x,gl_FragCoord.y-level))*Ir_01;
    Denominator+=(1.0/8.0)*weight(gl_FragCoord.xy,vec2(gl_FragCoord.x,gl_FragCoord.y-level));

    vec4 Ir_02 = imageLoad(historyColorImages[1], ivec2(gl_FragCoord.x+level,gl_FragCoord.y-level));
    Numerator+=(1.0/16.0)*weight(gl_FragCoord.xy,vec2(gl_FragCoord.x+level,gl_FragCoord.y-level))*Ir_02;
    Denominator+=(1.0/16.0)*weight(gl_FragCoord.xy,vec2(gl_FragCoord.x+level,gl_FragCoord.y-level));

    vec4 Ir_10 = imageLoad(historyColorImages[1], ivec2(gl_FragCoord.x-level,gl_FragCoord.y));
    Numerator+=(1.0/8.0)*weight(gl_FragCoord.xy,vec2(gl_FragCoord.x-level,gl_FragCoord.y))*Ir_10;
    Denominator+=(1.0/8.0)*weight(gl_FragCoord.xy,vec2(gl_FragCoord.x-level,gl_FragCoord.y));

    vec4 Ir_11 = imageLoad(historyColorImages[1], ivec2(gl_FragCoord.xy));
    Numerator+=(1.0/4.0)*weight(gl_FragCoord.xy,vec2(gl_FragCoord.xy))*Ir_11;
    Denominator+=(1.0/4.0)*weight(gl_FragCoord.xy,vec2(gl_FragCoord.xy));

    vec4 Ir_12 = imageLoad(historyColorImages[1], ivec2(gl_FragCoord.x+level,gl_FragCoord.y));
    Numerator+=(1.0/8.0)*weight(gl_FragCoord.xy,vec2(gl_FragCoord.x+level,gl_FragCoord.y))*Ir_12;
    Denominator+=(1.0/8.0)*weight(gl_FragCoord.xy,vec2(gl_FragCoord.x+level,gl_FragCoord.y));

    vec4 Ir_20 = imageLoad(historyColorImages[1], ivec2(gl_FragCoord.x-level,gl_FragCoord.y+level));
    Numerator+=(1.0/16.0)*weight(gl_FragCoord.xy,vec2(gl_FragCoord.x-level,gl_FragCoord.y+level))*Ir_20;
    Denominator+=(1.0/16.0)*weight(gl_FragCoord.xy,vec2(gl_FragCoord.x-level,gl_FragCoord.y+level));

    vec4 Ir_21 = imageLoad(historyColorImages[1], ivec2(gl_FragCoord.x,gl_FragCoord.y+level));
    Numerator+=(1.0/8.0)*weight(gl_FragCoord.xy,vec2(gl_FragCoord.x,gl_FragCoord.y+level))*Ir_21;
    Denominator+=(1.0/8.0)*weight(gl_FragCoord.xy,vec2(gl_FragCoord.x,gl_FragCoord.y+level));

    vec4 Ir_22 = imageLoad(historyColorImages[1], ivec2(gl_FragCoord.x+level,gl_FragCoord.y+level));
    Numerator+=(1.0/16.0)*weight(gl_FragCoord.xy,vec2(gl_FragCoord.x+level,gl_FragCoord.y+level))*Ir_22;
    Denominator+=(1.0/16.0)*weight(gl_FragCoord.xy,vec2(gl_FragCoord.x+level,gl_FragCoord.y+level));

    //vec4 Ir=(1/4.0)*Ir_11+(1/8.0)*(Ir_01+Ir_10+Ir_12+Ir_21)+(1/16.0)*(Ir_00+Ir_02+Ir_20+Ir_22);

    return Numerator/Denominator;
}