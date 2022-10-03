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

int gcd(int m,int n)    //求最大公因数
{
    int l=1;
    if(m<=n)
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

int BigPowerMod(int num,int power,int mod)
{
    int ans=1;
    num=num%mod;
    while(power>0)
    {
        if(power%2==1)
        {
            ans=(ans*num)%mod;
        }
        power/=2;
        num=num*num%mod;
    }
    return ans;
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
    int encode,decode;
    encode= BigPowerMod(num,e,n);
    cout<<"加密后结果为："<<encode<<endl;
    decode= BigPowerMod(encode,d,n);
    cout<<"解密后结果为："<<decode<<endl;
}
