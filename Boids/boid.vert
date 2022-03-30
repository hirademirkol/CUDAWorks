#version 330 core
uniform mat4 projectionMat;
uniform mat4 modelViewMat;
uniform float scale;

layout (location = 0) in vec2 aPos;
layout (location = 1) in vec3 offset;
out vec4 vColor;

void main()
{
    vec3 pos = scale * vec3(aPos, 0.0) + offset;
    gl_Position = projectionMat * modelViewMat * vec4(pos, 1.0);

    vColor = vec4(pos*0.5 + 0.5, 1.0);
}