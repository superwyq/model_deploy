#include<stdio.h>

void say_no(){
    printf("No,you are too close\n");
}

int up(int x){
    return x+1;
}

int main(){
    int begin = 0;
    int boundry = 10;
    printf("please input begin number:");
    scanf("%d", &begin);
    while(1){
        printf("begin = %d\n", begin);
        getchar();
        if (begin>boundry)
        {
            say_no();
        }
        begin = up(begin);
    }
}