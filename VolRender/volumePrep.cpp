#include <iostream>
#include <math.h>
#include <stdlib.h>

unsigned char f(int x, int y, int z);
void applyNoise(unsigned char *data);

unsigned char data[32*32*32];

int main(int argc, char **argv)
{
    typedef unsigned char VolumeType;

    FILE *fp = fopen((char*)"data/custom.raw", "w");
    size_t size = 32 * 32 * 32 * sizeof(VolumeType);


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

    applyNoise(data);

    size_t wrote = fwrite((void*)data, 1, size, fp);
    fclose(fp);
    
    return 0;
}

unsigned char f(int x, int y, int z)
{
    float xf = (float)x/16.0f;
    float yf = (float)y/16.0f;
    float zf = (float)z/16.0f;
    float r = 1.0f;
    float r_sq = r*r;

    //Soft sphere with r
    float w =(xf*xf + yf*yf + zf*zf)/ r_sq;
    // Clamp to 0-1
    w = w < 0.0f ? 0.0f : w;
    w = w > 1.0f ? 1.0f : w;
    //Inverse and square
    w = (1-w);
    w = sqrt(w);

    return (unsigned char)(255*w);
}

void applyNoise(unsigned char *data)
{
    float a, b;
    for(int x = 0; x < 31; x++)
    {
        for(int y = 0; y < 31; y++)
        {
            for(int z = 0; z < 31; z++)
            {
                a = (float)(rand() % 127) / 256;
                data[z*1024 + y*32 + x] = 0.5f*data[z*1024 + y*32 + x] + a*data[z*1024 + y*32 + x + 1] + (1-a)*data[z*1024 + y*32 + x + 32];
            }
        }
    }
}