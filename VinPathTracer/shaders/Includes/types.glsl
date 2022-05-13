#extension GL_EXT_control_flow_attributes : require

#define M_PI 3.1415926535897932384626433832795

struct Vertex{
  vec3 pos;
  vec3 normal;
  vec3 color;
  vec2 texCoord;
};

struct Primitive {
    uint material_id;
    int diffuse_idx;
};

struct Material {
  vec3 ambient;
  vec3 diffuse;
  vec3 specular;
  vec3 emission;
  int diffuse_idx;
};

struct QuadArealignt {         
    vec4 A;                    //    A* * *B
    vec4 B;                    //    *     *
    vec4 C;                    //    *     *
    vec4 D;                    //    C* * *D
};





