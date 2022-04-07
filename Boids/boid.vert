#version 330 core
uniform mat4 projectionMat;
uniform mat4 modelViewMat;
uniform float scale;

layout (location = 0) in vec2 vPos;
layout (location = 1) in vec4 boidPos;
layout (location = 2) in vec4 vel;
layout (location = 3) in vec4 up;

out vec3 fPos;
out vec3 fNormal;
out vec4 fColor;

void main()
{
    // Side vector of plane to project instance vertices to
    vec3 crossVec = cross(normalize(vel.xyz), normalize(up.xyz));
    
    fPos = scale * (vPos.x*crossVec + vPos.y*normalize(vel.xyz)) + boidPos.xyz;
    
    // Calculate a normal as vectors looking outwards from the triangle
    fNormal = vec3(vPos, 1.0f);
    
    gl_Position = projectionMat * modelViewMat * vec4(fPos, 1.0);
    fColor = vec4(abs(fPos), 1.0f);
}