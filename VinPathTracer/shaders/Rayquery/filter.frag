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
layout (binding = 8, rgba32f) uniform image2D historyColorImages[];
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

void main() {
    vec3 directColor = vec3(0.0, 0.0, 0.0);
    vec3 indirectColor = vec3(0.0, 0.0, 0.0);
    vec3 surfaceColor=vec3(0.0,0.0,0.0);
    vec4 historyColor=imageLoad(historyColorImages[1],ivec2(gl_FragCoord.xy));
   
    vec2 myFragCoord=getFragCoord(ubo.proj * ubo.view * ubo.model,interpolatedPosition);
    if(myFragCoord.x==gl_FragCoord.x) outDirectIr=vec4(0.5,0.0,0.0,1.0);
    else outDirectIr=vec4(0.0,0.5,0.0,1.0);

    outColor = historyColor;
    //outDirectIr=vec4(0.5,0.0,0.0,1.0);

    //if(isLightSource(materialBuffer.data[material_id].emission)) outColor = vec4(materialBuffer.data[material_id].emission,1.0f);
}