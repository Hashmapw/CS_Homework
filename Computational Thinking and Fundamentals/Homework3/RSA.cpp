#include<bits/stdc++.h>
using namespace std;

bool judgePrime(int n)
{
    if(n<=1)  return false;
    for(int i=2;i< sqrt(n)+1;i++)
    {
        if(n%i==0)  return false;
    }
    return true;
}

int main()
{
    int p,q;
    cin>>p;
    cout<<judgePrime(p);
}
