# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 23:34:35 2020

@author: Shaunak_Sensarma
"""
import math
print('Enter the number of terms to find golomb encoding')
n=int(input())
arr=[]
b=[]
print('Enter the numbers...')
for i in range(0,n):
    num=int(input())
    arr.append(num)
print('\nEnter b value which is a power of 2')
num=int(input())
b.append(num)
print('\nEnter b value which is not a power of 2')
num=int(input())
b.append(num)
print()
for i in range(0,n):
    x=arr[i]
    print("Number is",x)
    q=x//b[1]
    q=q+1
    r = x % b[1] 
    quo ='0'*(q-1)+'1'
    i = math.floor(math.log2(b[1])) 
    d = 2**(i + 1)-b[1] 

    if r < d: 
        rem = bin(r)[2:] 
        l = len(rem) 
        if l<i: 
            rem = '0'*(i-l)+rem 
    else: 
        rem = bin(r + d)[2:] 
        l = len(rem) 

        if l<i + 1: 
            rem = '0'*(b + 1-l)+rem 
    golomb_code = quo + rem 
    print("The Normal golomb encoding for x = {} and b = {} is {}". 
      format(x, b[1], golomb_code))
    
    q=x//b[0]
    quo ='0'*(q)+'1'
    r=x-(q*b[0])
    rem=bin(r).replace("0b", "")  
    golrise=quo+rem+""
    print("The golomb rise encoding for x = {} and b = {} is {}". 
      format(x, b[0], golrise))
    print()
    
def decode(x):
    num=0;
    for i in range(len(x)):
        num+=(int(x[len(x)-1-i])*(math.pow(2,i)));
    return num;

x=str(input('Enter code for golomb decoding : '))
x=list(x)
b=int(input('Enter value of b: '))
i=math.floor(math.log(b,2))
d=math.pow(2,i+1)-b


p2=0;
l=1;
print("The decoded values for the given encoded number are:")
while(p2<len(x)):
    t=0;
    flag=0;
    r=[];
    k=i;
    q=0;
    for p in range(p2,len(x)):
        if(x[p]=='0' and flag==0):
            t+=1;
            continue;
        if(x[p]=='1' and flag==0):
            q=t;
            flag=1;
            continue;
        r.append(x[p]);
        k-=1;
        if(k==0):
            rnum=decode(r);
            if(rnum<d):
                p2=p+1;
                break;
        if(k==-1):
            rnum=decode(r);
            rnum=rnum-d;
            p2=p+1;
            break;
    ans=q*b+rnum;
    ans=math.floor(ans)
    print(ans);
    l=0;
    