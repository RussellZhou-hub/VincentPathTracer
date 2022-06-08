#version 460
#extension GL_ARB_separate_shader_objects : enable
#extension GL_EXT_ray_tracing : enable
#extension GL_EXT_ray_query : enable
#extension GL_EXT_debug_printf : enable
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_nonuniform_qualifier: require

#include "../includes/types.glsl"

layout(binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
    mat4 prev_Proj_View;
    QuadArealignt qLight;
    uint frameCount;
     uint mode;  //denoising algorithm   1:raw  2:mvec 3:svgf 4:ours 5: ground truth
} ubo;

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColor;
layout(location = 2) in vec3 normal;
layout(location = 3) in vec2 inTexCoord;

layout(location = 0) out vec3 fragColor;
layout(location = 1) out vec3 fragNormal;
layout(location = 2) out vec3 interpolatedPosition;
layout(location = 3) out vec2 fragTexCoord;

void main() {
    interpolatedPosition=inPosition;
    gl_Position = ubo.proj * ubo.view * ubo.model * vec4(inPosition, 1.0);
    fragColor = inColor;
    fragNormal=normal;
    fragTexCoord = inTexCoord;
}