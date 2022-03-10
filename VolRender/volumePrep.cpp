#include <iostream>
#include <math.h>

unsigned char f(int x, int y, int z);

int main(int argc, char **argv)
{
    typedef unsigned char VolumeType;

    FILE *fp = fopen((char*)"data/custom.raw", "w");
    size_t size = 32 * 32 * 32 * sizeof(VolumeType);

    unsigned char data[32*32*32];

    for(int x = 0; x < 32; x++)
    {
        for(int y = 0; y < 32; y++)
        {
            for(int z = 0; z < 32; z++)
            {
                data[z*1024 + y*32 + x] = f(x-16, y-16, z-16);
            }
        }
    }

    size_t wrote = fwrite((void*)data, 1, size, fp);
    fclose(fp);
    
    return 0;
}

unsigned char f(int x, int y, int z)
{
    float xf = (float)x/32.0f;
    float yf = (float)y/32.0f;
    float zf = (float)z/32.0f;
    float r = 0.01f;
    float r_sq = r*r;

    //Soft sphere with r
    float w =(xf*xf - yf*yf - zf*zf)/ r_sq;
    // Clamp to 0-1
    w = w < 0.0f ? 0.0f : w;
    w = w > 1.0f ? 1.0f : w;
    //Inverse and square
    //w = (1-w);
    //w = sqrt(w);

    return (unsigned char)255*w;
}