#include <stdio.h>

#define element_type float
#define OFFSET(channel,element_index,matrix_num)((channel)*(matrix_num)+element_index) //计算矩阵的偏移量
//矩阵存储的本质是一个一维数组，所以需要通过偏移量来访问矩阵的元素
//存储方式是NCHW(也有NHWC的存储方式)

void cpu_convolution(element_type *in,element_type* out,element_type *kernel, int batch_size,
    int in_channel,int in_height,int in_width,
    int kernel_height,int kernel_width)
    {
        int out_channel=in_channel;
        int out_height=in_height-kernel_height+1;
        int out_width=in_width-kernel_width+1;
        element_type val;
        int out_pos,in_pos,kernel_pos;
        for(int oc=0;oc<out_channel;oc++){
            //对输出的每一个channel进行计算
            for (int i = 0; i < out_height; i++){ //行序，所以先遍历行
                for(int j=0;j<out_width;j++){
                    val=0;
                    out_pos=OFFSET(oc,i*out_width+j,out_height*out_width); //计算输出的偏移量
                    for(int ic=0;ic<in_channel;ic++){
                        //对输入的每一个channel进行计算,确保输出的每个channel都是由输入的所有channel计算得到
                        for(int ki=0;ki<kernel_height;ki++){ //卷积计算
                            for(int kj=0;kj<kernel_width;kj++){
                                in_pos=OFFSET(ic,i*kernel_height+ki,in_height*kernel_width)+kj;
                                kernel_pos=OFFSET(oc,ic,kernel_height*kernel_width);
                                val+=in[in_pos]*kernel[kernel_pos];
                            }
                        }
                    }
                    out[out_pos]=val;
                }
            }
            
        }
    }

int main(){
    element_type in[2*3*4*4];
    element_type out[2*2*2*2];
    element_type kernel[2*3*3*3];
    for(int i=0;i<2*3*4*4;i++){
        in[i]=i;
    }
    for(int i=0;i<2*2*2*2;i++){
        out[i]=0;
    }
    for(int i=0;i<2*3*3*3;i++){
        kernel[i]=i;
    }
    cpu_convolution(in,out,kernel,2,3,4,4,3,3);
    for(int i=0;i<2*2*2*2;i++){
        printf("%f ",out[i]);
    }
    return 0;
}