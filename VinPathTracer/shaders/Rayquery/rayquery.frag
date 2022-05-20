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
} ubo;

layout(binding = 1) uniform sampler2D texSampler;
layout (binding = 2) uniform sampler2D textures[];
layout(binding = 3) buffer MaterialIndexBuffer { Primitive data[]; } materialIndexBuffer;
layout(binding = 4) buffer MaterialBuffer { Material data[]; } materialBuffer;
layout(binding = 5,set=0) uniform accelerationStructureEXT topLevelAS;
layout(binding = 6) buffer VertexBuffer { Vertex data[]; } vertexBuffer;
layout(binding = 7) buffer IndexBuffer { uint data[]; } indexBuffer;

layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec3 fragNormal;
layout(location = 2) in vec3 interpolatedPosition;
layout(location = 3) in vec2 fragTexCoord;

layout(location = 0) out vec4 outColor;

void main() {
    vec3 directColor = vec3(0.0, 0.0, 0.0);
    vec3 indirectColor = vec3(0.0, 0.0, 0.0);
    vec3 surfaceColor=vec3(0.0,0.0,0.0);
    vec3 lightPos=get_Random_QuadArea_Light_Pos(ubo.qLight.A.xyz,  ubo.qLight.B.xyz,  ubo.qLight.C.xyz, ubo.qLight.D.xyz, ubo.frameCount);
    float irradiance=dot(fragNormal,vec3(lightPos-interpolatedPosition));
    irradiance=clamp(irradiance,0.0f,1.0f)/(0.001f*distance(lightPos,interpolatedPosition)+0.01f);
    vec4 diffuseColor;
    int diffuse_id=materialIndexBuffer.data[gl_PrimitiveID].diffuse_idx;
    uint material_id=materialIndexBuffer.data[gl_PrimitiveID].material_id;
    if(diffuse_id==-1){
        diffuseColor=vec4(materialBuffer.data[material_id].diffuse,1.0f);
    }
    else{
        diffuseColor=texture(textures[diffuse_id], fragTexCoord);
    }
    surfaceColor=diffuseColor.xyz;
    
    vec3 lightColor = vec3(0.6, 0.6, 0.6);
    vec3 lightPosition=lightPos;
    vec3 geometricNormal=fragNormal;

    vec3 positionToLightDirection = normalize(lightPosition - interpolatedPosition);

    vec3 shadowRayOrigin = interpolatedPosition;
    vec3 shadowRayDirection = positionToLightDirection;
    float shadowRayDistance = length(lightPosition - interpolatedPosition) - 0.001f;

    
    //shadow ray
    rayQueryEXT rayQuery;
    rayQueryInitializeEXT(rayQuery, topLevelAS, gl_RayFlagsTerminateOnFirstHitEXT| gl_RayFlagsOpaqueEXT | gl_RayFlagsSkipClosestHitShaderEXT, 0xFF, shadowRayOrigin, 0.001f, shadowRayDirection, shadowRayDistance);
  
    while (rayQueryProceedEXT(rayQuery));

    // If the intersection has hit a triangle, the fragment is shadowed
	if (rayQueryGetIntersectionTypeEXT(rayQuery, true) == gl_RayQueryCommittedIntersectionNoneEXT ) {
		directColor=surfaceColor * lightColor;// * dot(geometricNormal, positionToLightDirection); 
	}
    else{
        directColor=vec3(0.0,0.0,0.0);
    }

    
    ivec3 indices = ivec3(indexBuffer.data[3 * gl_PrimitiveID + 0], indexBuffer.data[3 * gl_PrimitiveID + 1], indexBuffer.data[3 * gl_PrimitiveID + 2]);
    
    vec3 vertexA =vertexBuffer.data[indices.x].pos;
    vec3 vertexB=vertexBuffer.data[indices.y].pos;
    vec3 vertexC=vertexBuffer.data[indices.z].pos;
    /*
    geometricNormal = normalize(cross(vertexB - vertexA, vertexC - vertexA));
    */
    vec3 rayOrigin = interpolatedPosition;
    vec3 rayDirection = getSampledReflectedDirection(ubo.cameraPos.xyz,interpolatedPosition.xyz,geometricNormal,gl_FragCoord.xy,ubo.frameCount);
    vec3 previousNormal = geometricNormal;
    float secondaryRayDistance = length(lightPosition - interpolatedPosition) - 0.001f;

    bool rayActive = true;
    int maxRayDepth = 1;
    if(true){
      for (int rayDepth = 0; rayDepth < maxRayDepth && rayActive; rayDepth++) {
        //secondary ray (or more ray)
        rayQueryEXT secondaryRayQuery;
        rayQueryInitializeEXT(secondaryRayQuery, topLevelAS, gl_RayFlagsTerminateOnFirstHitEXT, 0xFF, rayOrigin, 0.001f, rayDirection, 1000.0f);

        while (rayQueryProceedEXT(secondaryRayQuery));

        if (rayQueryGetIntersectionTypeEXT(secondaryRayQuery, true) != gl_RayQueryCommittedIntersectionNoneEXT) {
        
            int extensionPrimitiveIndex = rayQueryGetIntersectionPrimitiveIndexEXT(rayQuery, true);
            vec2 extensionIntersectionBarycentric = rayQueryGetIntersectionBarycentricsEXT(rayQuery, true);

            ivec3 extensionIndices = ivec3(indexBuffer.data[3 * extensionPrimitiveIndex + 0], indexBuffer.data[3 * extensionPrimitiveIndex + 1], indexBuffer.data[3 * extensionPrimitiveIndex + 2]);
            vec3 extensionBarycentric = vec3(1.0 - extensionIntersectionBarycentric.x - extensionIntersectionBarycentric.y, extensionIntersectionBarycentric.x, extensionIntersectionBarycentric.y);
      
            vec3 extensionVertexA =vertexBuffer.data[extensionIndices.x].pos;
            vec3 extensionVertexB=vertexBuffer.data[extensionIndices.y].pos;
            vec3 extensionVertexC=vertexBuffer.data[extensionIndices.z].pos;
    
            vec3 extensionPosition = extensionVertexA * extensionBarycentric.x + extensionVertexB * extensionBarycentric.y + extensionVertexC * extensionBarycentric.z;
            vec3 extensionNormal = normalize(cross(extensionVertexB - extensionVertexA, extensionVertexC - extensionVertexA));

            vec3 extensionSurfaceColor = materialBuffer.data[materialIndexBuffer.data[extensionPrimitiveIndex].material_id].diffuse;
            //extensionSurfaceColor = vec3(0.0,1.0,0.0);

            //vec2 RayHitPointFragCoord=getFragCoord(extensionPosition.xyz);

            
            int randomIndex = int(random(gl_FragCoord.xy, ubo.frameCount + rayDepth) * 2 + 40);
            vec3 lightColor = vec3(0.6, 0.6, 0.6);

            vec3 positionToLightDirection = normalize(lightPosition - extensionPosition);

            vec3 shadowRayOrigin = extensionPosition;
            vec3 shadowRayDirection = positionToLightDirection;
            float shadowRayDistance = length(lightPosition - extensionPosition) - 0.001f;
            
            rayQueryEXT rayQuery;
            rayQueryInitializeEXT(rayQuery, topLevelAS, gl_RayFlagsTerminateOnFirstHitEXT, 0xFF, shadowRayOrigin, 0.001f, shadowRayDirection, shadowRayDistance);
      
            while (rayQueryProceedEXT(rayQuery));//secondary shadow ray
            if (rayQueryGetIntersectionTypeEXT(rayQuery, true) == gl_RayQueryCommittedIntersectionNoneEXT) {
                indirectColor += (1.0 / (rayDepth + 1)) * extensionSurfaceColor * lightColor  * dot(previousNormal, rayDirection) * dot(extensionNormal, positionToLightDirection);
                //indirectColor=extensionSurfaceColor;
            }
            else {
                rayActive = false;
            }
            /*
            //vec3 hemisphere = uniformSampleHemisphere(vec2(random(gl_FragCoord.xy, ubo.frameCount + rayDepth), random(gl_FragCoord.xy, ubo.frameCount + rayDepth + 1)));
            //vec3 alignedHemisphere = alignHemisphereWithCoordinateSystem(hemisphere, extensionNormal);

            //reset rayOrigin...
            rayOrigin = extensionPosition;
            //rayDirection = alignedHemisphere;
            previousNormal = extensionNormal;

            //RayHitPointFragCoord=getFragCoord(interpolatedPosition.xyz);
            //RayHitPointFragCoord=getFragCoord(extensionPosition.xyz);
            */
            indirectColor=extensionSurfaceColor;
        }
        
        else {  //secondary ray not hit
            rayActive = false;
            //directColor=vec3(1.0,0.0,0.0);
        }
     }
   }
   
    outColor = vec4(directColor+indirectColor+diffuseColor.xyz*0.00f,1.0f);

    //if(isLightSource(materialBuffer.data[material_id].emission)) outColor = vec4(materialBuffer.data[material_id].emission,1.0f);
}