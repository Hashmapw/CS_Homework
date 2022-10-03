#include <bits/stdc++.h>
using namespace std;
//把矩阵直接置换为二进制
int S1[4][4][2]={{{0,1},{0,0},{1,1},{1,0}},
                 {{1,1},{1,0},{0,1},{0,0}},
                 {{0,0},{1,0},{0,1},{1,1}},
                 {{1,1},{0,1},{0,0},{1,0}}};
int S2[4][4][2]={{{0,0},{0,1},{1,0},{1,1}},
                 {{1,0},{0,0},{0,1},{1,1}},
                 {{1,1},{1,0},{0,1},{0,0}},
                 {{1,0},{0,1},{0,0},{1,1}}};

void CreateKey(int k[11],int k1[9],int k2[9])
{
    int temp[11];
    //P10置换
    temp[1]=k[3],temp[2]=k[5],temp[3]=k[2],temp[4]=k[7],temp[5]=k[4],
    temp[6]=k[10],temp[7]=k[1],temp[8]=k[9],temp[9]=k[8],temp[10]=k[6];
    int l[6],r[6];
    for(int i=1;i<=5;i++)
        l[i]=temp[i],r[i]=temp[i+5];
    int x1,x2,x3,x4;
    x1=l[1],x2=r[1];
    for(int i=2;i<=5;i++)
        l[i-1]=l[i],r[i-1]=r[i];
    l[5]=x1;
    r[5]=x2;
    for(int i=1;i<=5;i++)
        temp[i]=l[i],temp[i+5]=r[i];
    k1[1]=temp[6],k1[2]=temp[3],k1[3]=temp[7],k1[4]=temp[4],k1[5]=temp[8],k1[6]=temp[5],k1[7]=temp[10],k1[8]=temp[9];
    printf("密钥K1:");
    int ans=0;
    for(int i=1;i<=8;i++)
    {
        ans+=k1[i]*(1<<(8-i));
        printf("%d",k1[i]);
    }
    printf("\n");
    x1=l[1],x2=l[2],x3=r[1],x4=r[2];
    for(int i=2;i<=5;i++)
        l[i-2]=l[i],r[i-2]=r[i];
    l[4]=x1,l[5]=x2;
    r[4]=x3,r[5]=x4;
    for(int i=1;i<=5;i++)
        temp[i]=l[i],temp[i+5]=r[i];
    k2[1]=temp[6],k2[2]=temp[3],k2[3]=temp[7],k2[4]=temp[4],k2[5]=temp[8],k2[6]=temp[5],k2[7]=temp[10],k2[8]=temp[9];
    printf("密钥K2:");
    ans=0;
    for(int i=1;i<=8;i++)
    {
        ans+=k2[i]*(1<<(8-i));
        printf("%d",k2[i]);
    }
    printf("\n");
}

void f(int R[],int K[])
{
    int temp[9];
    temp[1]=R[4],temp[2]=R[1],temp[3]=R[2],temp[4]=R[3],temp[5]=R[2],temp[6]=R[3],temp[7]=R[4],temp[8]=R[1];

    for(int i=1;i<=8;i++)
        temp[i]=temp[i]^K[i];

    printf("异或结果:");
    int ans=0;
    for(int i=1;i<=8;i++)
    {
        ans+=temp[i]*(1<<(8-i));
        cout<<temp[i];
    }
    cout<<endl;

    int s1[5],s2[5];
    for(int i=1;i<=4;i++)
        s1[i]=temp[i],s2[i]=temp[i+4];
    int x1=S1[s1[1]*2+s1[4]][s1[2]*2+s1[3]][0],x2=S1[s1[1]*2+s1[4]][s1[2]*2+s1[3]][1];
    int x3=S2[s2[1]*2+s2[4]][s2[2]*2+s2[3]][0],x4=S2[s2[1]*2+s2[4]][s2[2]*2+s2[3]][1];

    //P4置换
    R[1]=x2,R[2]=x4,R[3]=x3,R[4]=x1;
    printf("P4置换结果:");
    ans=0;
    for(int i=1;i<=4;i++)
    {
        ans+=R[i]*(1<<(4-i));
        cout<<R[i];
    }
    cout<<endl;
}

void Encrypt(int text[9],int k1[],int k2[])
{
    int temp[9];
    //IP置换（IP={2,6,3,1,4,8,5,7}）
    temp[1]=text[2],temp[2]=text[6],temp[3]=text[3],temp[4]=text[1],
    temp[5]=text[4],temp[6]=text[8],temp[7]=text[5],temp[8]=text[7];
    printf("IP置换:");
    int ans=0;
    for(int i=1;i<=8;i++)
    {
        ans+=temp[i]*(1<<(8-i));
        cout<<temp[i];
    }
    cout<<endl;
    int L0[5],R0[5],L1[5],R1[5],L2[5],R2[5];
    for(int i=1;i<=4;i++)
        L0[i]=temp[i],R0[i]=temp[i+4];
    memcpy(L1,R0,sizeof(L1));

    //复合函数fk1
    printf("L0:");
    ans=0;
    for(int i=1;i<=4;i++)
    {
        ans+=L0[i]*(1<<(4-i));
        cout<<L0[i];
    }
    cout<<" ";

    printf("R0:");
    ans=0;
    for(int i=1;i<=4;i++)
    {
        ans+=R0[i]*(1<<(4-i));
        cout<<R0[i];
    }
    cout<<endl;
    f(R0,k1);
    for(int i=1;i<=4;i++)
        R1[i]=L0[i]^R0[i];

    //置换函数
    memcpy(R2,R1,sizeof(R2));

    printf("L1:");
    ans=0;
    for(int i=1;i<=4;i++)
    {
        ans+=L1[i]*(1<<(4-i));
        cout<<L1[i];
    }
    cout<<" ";;

    printf("R1:");
    ans=0;
    for(int i=1;i<=4;i++)
    {
        ans+=R1[i]*(1<<(4-i));
        cout<<R1[i];
    }
    cout<<endl;


    //复合函数fk2
    f(R1,k2);
    for(int i=1;i<=4;i++)
        L2[i]=L1[i]^R1[i];

    printf("L2:");
    ans=0;
    for(int i=1;i<=4;i++)
    {
        ans+=L2[i]*(1<<(4-i));
        cout<<L2[i];
    }
    cout<<" ";

    printf("R2:");
    ans=0;
    for(int i=1;i<=4;i++)
    {
        ans+=R2[i]*(1<<(4-i));
        cout<<R2[i];
    }
    cout<<endl;

    temp[1]=L2[4],temp[2]=L2[1],temp[3]=L2[3],temp[4]=R2[1],temp[5]=R2[3],temp[6]=L2[2],temp[7]=R2[4],temp[8]=R2[2];
    printf("加密结果:");
    ans=0;

    //IP-1置换
    for(int i=1;i<=8;i++)
    {
        ans+=temp[i]*(1<<(8-i));
        cout<<temp[i];
    }
    cout<<endl;
}

void Decrypt(int text[9],int k1[],int k2[])
{
    int temp[9];
    //IP置换（IP={2,6,3,1,4,8,5,7}）
    temp[1]=text[2],temp[2]=text[6],temp[3]=text[3],temp[4]=text[1],
    temp[5]=text[4],temp[6]=text[8],temp[7]=text[5],temp[8]=text[7];
    printf("IP置换:");
    int ans=0;
    for(int i=1;i<=8;i++)
    {
        ans+=temp[i]*(1<<(8-i));
        cout<<temp[i];
    }
    cout<<endl;
    int L0[5],R0[5],L1[5],R1[5],L2[5],R2[5];
    for(int i=1;i<=4;i++)
        L0[i]=temp[i],R0[i]=temp[i+4];
    memcpy(L1,R0,sizeof(L1));

    //复合函数fk1
    printf("L0:");
    ans=0;
    for(int i=1;i<=4;i++)
    {
        ans+=L0[i]*(1<<(4-i));
        cout<<L0[i];
    }
    cout<<" ";

    printf("R0:");
    ans=0;
    for(int i=1;i<=4;i++)
    {
        ans+=R0[i]*(1<<(4-i));
        cout<<R0[i];
    }
    cout<<endl;
    f(R0,k2);
    for(int i=1;i<=4;i++)
        R1[i]=L0[i]^R0[i];

    //置换函数
    memcpy(R2,R1,sizeof(R2));

    printf("L1:");
    ans=0;
    for(int i=1;i<=4;i++)
    {
        ans+=L1[i]*(1<<(4-i));
        cout<<L1[i];
    }
    cout<<" ";;

    printf("R1:");
    ans=0;
    for(int i=1;i<=4;i++)
    {
        ans+=R1[i]*(1<<(4-i));
        cout<<R1[i];
    }
    cout<<endl;


    //复合函数fk2
    f(R1,k1);
    for(int i=1;i<=4;i++)
        L2[i]=L1[i]^R1[i];

    printf("L2:");
    ans=0;
    for(int i=1;i<=4;i++)
    {
        ans+=L2[i]*(1<<(4-i));
        cout<<L2[i];
    }
    cout<<" ";;

    printf("R2:");
    ans=0;
    for(int i=1;i<=4;i++)
    {
        ans+=R2[i]*(1<<(4-i));
        cout<<R2[i];
    }
    cout<<endl;

    temp[1]=L2[4],temp[2]=L2[1],temp[3]=L2[3],temp[4]=R2[1],temp[5]=R2[3],temp[6]=L2[2],temp[7]=R2[4],temp[8]=R2[2];
    printf("解密结果:");
    ans=0;

    //IP-1置换
    for(int i=1;i<=8;i++)
    {
        ans+=temp[i]*(1<<(8-i));
        cout<<temp[i];
    }
    cout<<endl;
}


int main()
{
    int mode,plaintext[9],ciphertext[9];
    int k[11],k1[9],k2[9];
    printf("请输入主密钥:");
    for(int i=1;i<=10;i++)
    {
        scanf("%1d",&k[i]);
    }
    CreateKey(k,k1,k2);
    printf("模式菜单：【1】加密  【2】解密\n");
    printf("请选择模式：");
    scanf("%d",&mode);
    if(mode==1)
    {
        printf("请输入明文:");
        for(int i=1;i<=8;i++)
        {
            scanf("%1d",&plaintext[i]);
        }
        Encrypt(plaintext,k1,k2);
    }
    else if(mode==2)
    {
        printf("请输入密文：");
        for(int i=1;i<=8;i++)
        {
            scanf("%1d",&ciphertext[i]);
        }
        Decrypt(ciphertext,k1,k2);
    }
    else
    {
        printf("输入错误，请重新输入");
    }
    return 0;
}