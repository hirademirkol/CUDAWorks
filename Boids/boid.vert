#version 330 core
uniform mat4 projectionMat;
uniform mat4 modelViewMat;
uniform float scale;

layout (location = 0) in vec2 vPos;
layout (location = 1) in vec4 boidPos;
layout (location = 2) in vec4 vel;
layout (location = 3) in vec4 up;

out vec4 vColor;

void main()
{
    vec3 crossVec = cross(normalize(vel.xyz), normalize(up.xyz));
    vec3 pos = scale * (vPos.x*crossVec + vPos.y*normalize(vel.xyz)) + boidPos.xyz;
    gl_Position = projectionMat * modelViewMat * vec4(pos, 1.0);

    vColor = vec4(vec3(length(vel.xyz)),1.0);
}