#include<bits/stdc++.h>
using namespace std;

bool judgePrime(int n)    //�ж�����
{
    if(n<=1)  return false;
    if(n==2)  return true;
    for(int i=2;i< sqrt(n)+1;i++)
    {
        if(n%i==0)  return false;
    }
    return true;
}

int gcd(int m,int n)    //���������
{
    int l=1;
    if(m<=n)
    {
        cout<<"ѡ���e���ڵ���ŷ������\n";
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
        cout<<"��������������p.q��";
        cin>>p>>q;
        if(judgePrime(p)&& judgePrime(q))
            prime=true;
        else cout<<"�������������������\n";
    }
    n=p*q;
    m=(p-1)*(q-1);
    cout<<"ŷ��������ֵΪ��"<<m<<endl;
    prime=false;
    while(prime== false)
    {
        cout<<"��������ŷ�����������ұ�С��ŷ������������";
        cin>>e;
        if(gcd(m,e)==1)
        {
            prime=true;
            d=calculate_d(e,m);
            if(d==0)   cout<<"���ݷ�Χ��������"<<endl;
        }
    }
    cout<<"��Կ��"<<"e="<<e<<" "<<"n="<<n<<endl;
    cout<<"˽Կ��"<<"d="<<d<<" "<<"n="<<n<<endl;
    cout<<"����������Ҫ���ܵ�����";
    cin>>num;
    int encode,decode;
    encode= BigPowerMod(num,e,n);
    cout<<"���ܺ���Ϊ��"<<encode<<endl;
    decode= BigPowerMod(encode,d,n);
    cout<<"���ܺ���Ϊ��"<<decode<<endl;
}
