#include<bits/stdc++.h>
using namespace std;

string ToBinary(int n)   //其实可以调用itoa函数，但是需要额外的头文件支持
{
    string r;
    while (n != 0){
        r += ( n % 2 == 0 ? "0" : "1" );
        n /= 2;
    }
    return r;
}

int main()
{
    char letter;
    int asciiNum,countnum,mode;
    string asciiRe;
    cout<<"请输入ASCII码字符：";
    cin>>letter;
    asciiNum=(int)letter;
    asciiRe=ToBinary(asciiNum);
    string ascii(asciiRe.rbegin(),asciiRe.rend());
    cout<<"模式菜单：【1】奇校验 【2】偶校验\n";
    cout<<"请选择校验模式：";
    cin>>mode;
    if(mode==1)
    {
        countnum=count(ascii.begin(),ascii.end(),'1');
        if(countnum%2==0)   ascii+='1';
        else                ascii+='0';
        cout<<"奇校验后"<<letter<<"的编码为：";
    }
    else if(mode==2)
    {
        countnum=count(ascii.begin(),ascii.end(),'1');
        if(countnum%2==0)   ascii+='0';
        else                ascii+='1';
        cout<<"偶校验后"<<letter<<"的编码为：";
    }
    else
        cout<<"输入错误，请重新输入";
    for(int i=0;i<ascii.length();i++)
    {
        cout<<ascii[i];
    }
}
