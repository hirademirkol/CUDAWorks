#version 330 core
in vec3 fPos;
in vec3 fNormal;
in vec4 fColor;

out vec4 color;

void main()
{
        // Point Light Source
        const vec3 lightPos = vec3(0.577, 0.577, 0.577);
        vec3 lightDir = normalize(lightPos - fPos);

        //Getting absolute value for double siding
        float diff = abs(dot(normalize(fNormal), lightDir));

        // 0.2 Ambient Lighting
        color.rgb = fColor.rgb * (0.2f + 0.8f*diff);
        color.a = 1.0f;
}