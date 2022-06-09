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
    mat4 prev_Proj_View;
    QuadArealignt qLight;
    vec4 cameraPos;
    uint frameCount;
    uint mode;  //denoising algorithm   1:raw  2:mvec 3:svgf 4:ours 5: ground truth
    uint samples;  //spp
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
layout (binding = 10, rgba32f) uniform image2D historyDirectIr;
layout (binding = 11, rgba32f) uniform image2D historyIndIr;
layout (binding = 12, rgba32f) uniform image2D historyIndAlbedo;
layout (binding = 13, rgba32f) uniform image2D historyFinal;

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
vec4 next_itr_Var=vec4(0.0,0.0,0.0,0.0);

float weight(vec2 p,vec2 q);
float w_depth(vec2 p,vec2 q);
float w_normal(vec2 p,vec2 q);
float w_pos(vec2 p,vec2 q);
float w_lumin(vec2 p,vec2 q);
vec4 variance(vec2 p);
vec4 aTrous_directIr_5_5(vec2 p);
vec4 aTrous_indIr_5_5(vec2 p);
vec4 aTrous_indAlbedo_5_5(vec2 p);

void main() {
    vec3 directColor = vec3(0.0, 0.0, 0.0);
    vec3 indirectColor = vec3(0.0, 0.0, 0.0);
    vec3 surfaceColor=vec3(0.0,0.0,0.0);
    vec4 direcIr=vec4(0.0,0.0,0.0,0.0);
    vec4 indIr=vec4(0.0,0.0,0.0,0.0);

    if(ubo.mode==OURS){
        outIndAlbedo=aTrous_indAlbedo_5_5(gl_FragCoord.xy);
        imageStore(historyIndAlbedo, ivec2(gl_FragCoord.xy),outIndAlbedo);
    }
    else{
        outIndAlbedo=imageLoad(historyIndAlbedo, ivec2(gl_FragCoord.xy));
        imageStore(historyIndAlbedo, ivec2(gl_FragCoord.xy),outIndAlbedo);
    }

    if(ubo.mode==3 ||ubo.mode==4){
            direcIr=aTrous_directIr_5_5(gl_FragCoord.xy);
            indIr=aTrous_indIr_5_5(gl_FragCoord.xy);
            outDirectIr=0.7*direcIr;
            imageStore(historyColorImages[1], ivec2(gl_FragCoord.xy),direcIr);
            imageStore(historyDirectIr, ivec2(gl_FragCoord.xy),direcIr);
            imageStore(historyColorImages[3], ivec2(gl_FragCoord.xy),indIr);
            outIndIr=0.6*indIr;
    }
    else{
        direcIr=imageLoad(historyDirectIr, ivec2(gl_FragCoord.xy));
        indIr=imageLoad(historyColorImages[3], ivec2(gl_FragCoord.xy));
        outDirectIr=direcIr;
        outIndIr=indIr;
        imageStore(historyDirectIr, ivec2(gl_FragCoord.xy),direcIr);
        imageStore(historyColorImages[3], ivec2(gl_FragCoord.xy),indIr);
        
    }   

    //next_itr_Var=vec4(0.0,0.0,variance(gl_FragCoord.xy).z,1.0);
    //imageStore(historyColorImages[6],ivec2(gl_FragCoord.xy),next_itr_Var);
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
    float sigma_n=32;
    vec3 n_p=imageLoad(historyColorImages[4],ivec2(p.xy)).xyz;
    vec3 n_q=imageLoad(historyColorImages[4],ivec2(q.xy)).xyz;
    float weight=pow(max(0,dot(n_p,n_q)),sigma_n);
    return weight;
    //return 1.0;
}

float w_pos(vec2 p,vec2 q){   //weight of pos in the edge stop function add by me
    float sigma_x=0.1;
    float sigma_p=32;
    float epsil=0.0001;
    vec4 x_p=imageLoad(historyColorImages[5],ivec2(p.xy));
    vec4 x_q=imageLoad(historyColorImages[5],ivec2(q.xy));
    float weight=min(1.0f,exp(-distance(x_p.xyz,x_q.xyz)/(sigma_p+epsil)));
    float weight_ID=min(1.0f,exp(-sigma_p*distance(x_p.w,x_q.w)/(sigma_x+epsil)));
    if(ubo.mode==3) return weight;
    if(ubo.mode==4) return weight*weight_ID;
}

float w_lumin(vec2 p,vec2 q){//weight of Luminance in the edge stop function in SVGF
    float sigma_l=100;
    float epsil=0.01;
    float lumin_p=length(imageLoad(historyDirectIr,ivec2(p.xy)).xyz);
    float lumin_q=length(imageLoad(historyDirectIr,ivec2(q.xy)).xyz);
    float diff=abs(distance(lumin_p,lumin_q)/SQRT_OF_TREE);
    vec4 var=variance(p);
    float weight=exp(-diff/(sigma_l*sqrt(var).z+epsil));
    return weight;
}

vec4 variance(vec2 p){
    vec4 var=imageLoad(historyColorImages[6],ivec2(p));
    return var;
}

float weight(vec2 p,vec2 q){
     return  w_normal(p,q)*w_lumin(p,q)*w_pos(p,q);//w_depth(p,q)*
}

vec4 aTrous_directIr_5_5(vec2 p){
    vec4 Numerator=vec4(0.0,0.0,0.0,1.0);
    vec4 Denominator=vec4(0.0,0.0,0.0,1.0);
    vec4 Numerator_var=vec4(0.0,0.0,0.0,1.0);
    float level=2;
    for(int i=-2;i<=2;i++){
        for(int j=-2;j<=2;j++){
            int h_idx=5*(2+j)+2+i;
            float w=weight(gl_FragCoord.xy,vec2(gl_FragCoord.x+i*level,gl_FragCoord.y+j*level));
            Numerator+=h[h_idx]*w*imageLoad(historyDirectIr, ivec2(gl_FragCoord.x+i*level,gl_FragCoord.y+j*level));
            Denominator+=h[h_idx]*w;
            //Numerator_var+=h[h_idx]*h[h_idx]*w*w*variance(vec2(gl_FragCoord.x+i*level,gl_FragCoord.y+j*level));
        }
    }
    Numerator_var/=Denominator*Denominator;
    //next_itr_Var=Numerator_var;

    vec4 outTrous=Numerator/Denominator;
    outTrous.w=imageLoad(historyDirectIr, ivec2(gl_FragCoord.xy)).w;

    return outTrous;
}

vec4 aTrous_indIr_5_5(vec2 p){
    vec4 Numerator=vec4(0.0,0.0,0.0,1.0);
    vec4 Denominator=vec4(0.0,0.0,0.0,1.0);
    float level=1;
    for(int i=-2;i<=2;i++){
        for(int j=-2;j<=2;j++){
            int h_idx=5*(2+j)+2+i;
            float w=weight(gl_FragCoord.xy,vec2(gl_FragCoord.x+i*level,gl_FragCoord.y+j*level));
            Numerator+=h[h_idx]*w*imageLoad(historyIndIr, ivec2(gl_FragCoord.x+i*level,gl_FragCoord.y+j*level));
            Denominator+=h[h_idx]*w;
        }
    }

    vec4 outTrous=Numerator/Denominator;
    outTrous.w=1.0;

    return outTrous;
}

vec4 aTrous_indAlbedo_5_5(vec2 p){
    vec4 Numerator=vec4(0.0,0.0,0.0,1.0);
    vec4 Denominator=vec4(0.0,0.0,0.0,1.0);
    float level=1;
    for(int i=-2;i<=2;i++){
        for(int j=-2;j<=2;j++){
            int h_idx=5*(2+j)+2+i;
            float w=weight(gl_FragCoord.xy,vec2(gl_FragCoord.x+i*level,gl_FragCoord.y+j*level));
            Numerator+=h[h_idx]*w*imageLoad(historyIndAlbedo, ivec2(gl_FragCoord.x+i*level,gl_FragCoord.y+j*level));
            Denominator+=h[h_idx]*w;
        }
    }

    vec4 outTrous=Numerator/Denominator;
    outTrous.w=1.0;

    return outTrous;
}