#include<bits/stdc++.h>
using namespace std;

bool judgePrime(int n)    //判断素数
{
    if(n<=1)  return false;
    if(n==2)  return true;
    for(int i=2;i< sqrt(n)+1;i++)
    {
        if(n%i==0)  return false;
    }
    return true;
}

int gcd(int m,int n)    //求最大公因数   m大于n
{
    int l=1;
    if(m<n)
    {
        cout<<"选择的e大于等于欧拉函数\n";
        return 0;
    }
    while(m % n !=0){
        l = m% n;
        m = n;
        n = l;
    }
    return l;
}

int calculate_d(int e,int m)
{
    for(int k=1;;k++)
    {
        for(int d=1;d<(k*m)/e+2;d++)
        {
            if(e*d==k*m+1) {
                return d;
            }
        }
    }
    return 0;
}

int BigMod(int num, int power, int mod)
{
    long long result = num;
    long long temp = 1;

    while (power > 0) {
        if (power && 1) {
            // 如果幂为奇数，再乘一次
            temp = (temp % mod) * (result % mod) % mod;
        }
        result = (result % mod) * (result % mod) % mod;
        power /= 2;
    }
    return result;
}


int main()
{
    int p,q,e,d,n,m,num;
    bool prime=false;
    while(prime== false)
    {
        cout<<"请输入两个质数p.q：";
        cin>>p>>q;
        if(judgePrime(p)&& judgePrime(q))
            prime=true;
        else cout<<"输入的两个数不是素数\n";
    }
    n=p*q;
    m=(p-1)*(q-1);
    cout<<"欧拉函数的值为："<<m<<endl;
    prime=false;
    while(prime== false)
    {
        cout<<"请输入与欧拉函数互素且比小于欧拉函数的数：";
        cin>>e;
        if(gcd(m,e)==1)
        {
            prime=true;
            d=calculate_d(e,m);
            if(d==0)   cout<<"数据范围存在问题"<<endl;
        }
    }
    cout<<"公钥："<<"e="<<e<<" "<<"n="<<n<<endl;
    cout<<"私钥："<<"d="<<d<<" "<<"n="<<n<<endl;
    cout<<"请输入你想要加密的数：";
    cin>>num;
    unsigned long long int encode,decode;
    encode= BigMod(m,e,n);
    cout<<"加密后结果为"<<encode<<endl;
    decode= BigMod(encode,d,n);
    cout<<"解密后结果为"<<decode<<endl;
}
