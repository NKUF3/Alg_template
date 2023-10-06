#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
//快速幂 用于计算 a^b%mod
ll fpow(ll a,ll b,ll mod){
    a%=mod;
    ll res = 1;
    while(b){
        if(b&1)
            res=res*a%mod;
        a=a*a%mod;
        b>>=1;
    }
    return res;
}
//快速乘法1 复杂度O（logb）用于计算 a*b%mod
ll fastMul(ll a,ll b,ll mod){
    a%=mod;
    ll ans=0;
    while(b>0){
        if(b&1)ans=(ans+a)%mod;
        b>>=1;
        a=(a+a)%mod;
    }
    return ans;
}
//快速乘法2 使用long double 用于计算 a*b%mod
ll modMul(ll a,ll b,ll mod){
    if (mod<=1000000000) return a*b%mod;
    else if (mod<=1000000000000ll) return (((a*(b>>20)%mod)<<20)+(a*(b&((1<<20)-1))))%mod;
    else {
        ll d=(ll)floor(a*(long double)b/mod+0.5);
        ll ret=(a*b-d*mod)%mod;
        if (ret<0) ret+=mod;
        return ret;
    }
}
//快速除法 需要fpow fastMul/modMul 用于计算a/b%mod 需保证mod是素数 eg.998244353
ll fastDiv(ll a,ll b,ll mod){
    return fastMul(a,fpow(b,mod-2,mod),mod);
}
//若不能保证是素数 备选方案 保证a是b的倍数
ll Div2(ll a,ll b,ll mod){
    return a%(b*mod)/b;
}
//最大公因数 最小公倍数
ll gcd(ll a, ll b){
    return b?gcd(b,a%b):a;
}
ll lcm(ll a, ll b){
    return a/gcd(a,b)*b;
}
//扩展欧几里得 求解gcd的同时求解 ax+by=d的一组特解
ll ext_gcd(ll a,ll b,ll& x,ll& y){
    ll d = a;
    if (!b){
        x = 1;y = 0;
    }else{
        d = ext_gcd(b,a%b,y,x);
        y -= a/b*x;
    }
    return d;
}
//ax+by=c ax≡c(mol b) 通解为 x=c/gcd*x0+b/gcd*t 保证c%d=0 if a b 互质 取c=1求逆元
ll cal(ll a,ll b,ll c){
    ll x,y;
    ll gcd=ext_gcd(a,b,x,y);
    if(c%gcd)return -1;
    //if(gcd<0){gcd=-gcd;x=-x;} //在不能保证a b为正数时加入特判
    return ((x*c/gcd)%(b/gcd)+b/gcd)%(b/gcd);
}
//中国剩余定理 x≡a1(mol m1)...方程组 求解最小x (a1...an互质)
ll Sunzi(ll *m,ll *a,int len){
    ll lcm = 1;
    ll ans = 0;
    for (int i=0;i<len;i++){
        ll k0,ki;
        ll d = ext_gcd(lcm,m[i],k0,ki);
        if ((a[i]-ans)%d!=0) return -1;
        else {
            ll t = m[i]/d;
            k0 = ( k0*(a[i]-ans)/d%t + t)%t;
            ans = k0*lcm + ans;
            lcm = lcm/d*m[i];
        }
    }
    return ans;
}
//素数判断1
bool isPrime1(ll n){
    if(n==1)return false;
    for(ll i=2;i*i<=n;i++)
        if(n%i==0)return false;
    return true;
}
//素数判断2
bool isPrime2(ll n){
    if(n==2||n==3||n==5)return 1;
    if(n%2==0||n%3==0||n%5==0||n==1)return 0;
    ll c=7,a[8]={4,2,4,2,4,6,2,6};
    while(c*c<=n)for(auto i:a){if(n%c==0)return 0;c+=i;}
    return 1;
}
//素数判断3
class isprime3{
public:
    ll Rand(){
    static ll x=(srand((int)time(0)),rand());
    x+=1000003;
    if(x>1000000007)x-=1000000007;
    return x;
    }
    bool Witness(ll a,ll n){
        ll t=0,u=n-1;
        while(!(u&1))u>>=1,t++;
        ll x=fpow(a,u,n),y;
        while(t--){
            y=x*x%n;
            if(y==1 && x!=1 && x!=n-1)return true;
            x=y;
        }
        return x!=1;
    }
    bool MillerRabin(ll n,ll s){
        if(n==2||n==3||n==5)return 1;
        if(n%2==0||n%3==0||n%5==0||n==1)return 0;
        while(s--){
            if(Witness(Rand()%(n-1)+1,n))return false;
        }
        return true;
    }
    bool ans;//素数和查找次数
    isprime3(ll n,ll t){
        ans=MillerRabin(n,t);
    }
};//input:isprime3 ask(n,t);cout<<ask.ans;
//素数筛1 空间复杂度较小 时间复杂度nlogn 需要bool数组存储是否为质数
void getPrime(bool p[],int n) {
    for (int i = 1; i <= n; i++)p[i] = true;
    p[1] = false;
    for (int i = 2; i <= n; i++) {
        if (p[i]) {
            for (int j = i + i; j <= n; j += i)p[j] = false;
        }
    }
}
//素数筛2 时间复杂度n 空间复杂度更大 需要两个数组 可生成素数队列
ll getPrime(ll n,bool vis[],ll prime[]){
    ll tot=0;
    for(ll i=1;i<=n;i++)vis[i]=0;
    for(ll i=2;i<=n;i++){
        if(!vis[i])prime[tot++]=i;
        for(ll j=0;j<tot;j++){
            if(prime[j]*i>n)break;
            vis[prime[j]*i]=1;
            if(i%prime[j]==0)break;
        }
    }
    return tot;
}
//组合数
ll C(ll n,ll m){//n>m
    return (ll)round(tgamma(n+1)/tgamma(m+1)/tgamma(n-m+1));
}//tgamma（n+1)=n!
//组合数取余1 n m不大预处理
const ll mo=6662333;
ll c[1005][1005];
void getC(int n){
    for(int i=0;i<=n;i++){
        for(int j=0;j<=i;j++){
            if(j==0 || j==i)
                c[i][j]=1;
            else
                c[i][j]=(c[i-1][j-1]+c[i-1][j])%mo;
        }
    }
}
//组合数取余2 求逆元
//const ll mo=1e9+7
const ll N=1e7+10;
ll fastC(ll n,ll m) {
    static ll M = 0, inv[N], mul[N], invMul[N];
    while (M <= n) {
        if (M) {
            inv[M] = M == 1 ? 1 : (mo - mo / M) * inv[mo % M] % mo;
            mul[M] = mul[M - 1] * M % mo;
            invMul[M] = invMul[M - 1] * inv[M] % mo;
        } else mul[M] = 1, invMul[M] = 1;
        M++;
    }
    return mul[n] * invMul[m] % mo * invMul[n - m] % mo;
}
//组合数取余3
ll Lucas(ll n,ll m,ll p){//n>m
    ll ans=1;
    while(n|m)ans=ans*fastC(n%p,m%p)%p,n/=p,m/=p;
    return ans;
}
//pollard rho求因数 !有概率输出本身
inline ll PR(ll x)
{
    ll s=0,t=0,c=1ll*rand()%(x-1)+1;
    int stp=0,goal=1;
    ll val=1;
    for(goal=1;;goal<<=1,s=t,val=1)
    {
        for(stp=1;stp<=goal;++stp)
        {
            t=((__int128)t*t+c)%x;
            val=(__int128)val*abs(t-s)%x;
            if((stp%127)==0)
            {
                ll d=gcd(val,x);
                if(d>1)
                    return d;
            }
        }
        ll d=gcd(val,x);
        if(d>1)
            return d;
    }
}
//bsgs 求解A^x≡B(mol P) 需要a和p互质
inline ll BSGS(ll a, ll b, ll p)
{
    ll t = ceil(sqrt(p));
    std::map<ll, ll> hash;
    //map实现hash表
    hash.clear();
    ll tmp = b;
    for (int i = 1; i <= t; ++i)
    {
        tmp = tmp * a % p;
        hash[tmp] = i;
    }
    //插入b*a^j
    ll pw = fpow(a, t, p);
    tmp = pw;
    for (ll i = 1; i <= t; ++i)
    {
        if (hash.find(tmp) != hash.end())
            return i * t - hash[tmp];
        tmp = tmp * pw % p;
    }//查询a^(it)
    return -1; //返回无解
}
//不满足互质的补充 虽然在unknown error 但是原题主的ac代码也会unknownerror 应该没啥问题（
inline ll check(ll a,ll b,ll p) {
    ll k = 1 % p;
    for (int i = 0; i <= 40; ++i) {
        if (k == b)
            return i;
        k = k * a % p;
    }
    if (!a)
        return -1;
    return 0xfffffffff;
}
ll BSGSex(ll a,ll b,ll p){
    a %= p, b %= p;
    if (check(a,b,p)!=0xfffffffff)
        return check(a,b,p);
    ll d;
    ll ap = 1, n = 0;
    bool flg = false;
    while ((d = std::__gcd(a, p)) != 1){
        ++n;
        ap = ap * (a / d) % p;
        p /= d;
        if (b % d){
            flg = true;
            break;
        }
        b /= d;
    }
    if (flg)
        return -1;
    else
    {
        ll res = BSGS(a, 1LL * b * cal(ap, p, 1) % p, p);
        if (res == -1)
            return -1;
        else
            return res + n;
    }
}
//求原根 前置get_eulers(N);init();
vector<ll> primes;
bitset<N> st, exist;
ll euler[N];
void get_eulers(ll n) {
    euler[1] = 1;
    for (ll i = 2; i <= n; i++) {
        if (!st[i]) primes.push_back(i), euler[i] = i - 1;
        for (ll j = 0; primes[j] <= n / i; j++) {
            st[i * primes[j]] = 1;
            if (i % primes[j] == 0) {
                euler[i * primes[j]] = euler[i] * primes[j];
                break;
            }
            euler[i * primes[j]] = euler[i] * (primes[j] - 1);
        }
    }
}
void init() {//标记是否存在原根
    exist[1] = exist[2] = exist[4] = 1;
    for (auto x : primes) {
        if (x % 2) {
            for (ll i = x; i < N; i *= x) {
                exist[i] = 1;
                if (i * 2 < N) exist[i * 2] = 1;
            }
        }
    }
}
vector<ll> get_primesfactor(ll x) {//分解质因数
    vector<ll> res;
    for (auto prime : primes) {
        if (prime > x) break;
        if (x % prime == 0) res.push_back(prime);
    }
    return res;
}
ll primitiveroot(ll m) {
    //vector<ll> primitiveroot(ll m) {求所有原根使用
    //vector<ll> v;
    ll v;
    if (!exist[m]) return -1;//return v
    ll phi = euler[m], fst;
    auto factors = get_primesfactor(phi);
    for (ll i = 1;; i++) {
        if (gcd(i, m) != 1) continue;
        bool ok = true;
        for (auto x : factors) {
            if (fpow(i, phi / x, m) == 1) {
                ok = false;
                break;
            }
        }
        if (ok) {
            v = i;//fst=i
            break;
        }
    }
    /*ll cur = fst;
    for (ll i = 1; i <= phi; i++) {
        if (gcd(phi, i) == 1) v.push_back(cur);
        cur = cur * fst % m;
    }*/
    return v;
}
//考虑求解x^a≡b(mol P) 高次同余方程
ll ModEquationSolve(ll a,ll b,ll p){
    a%=p-1;
    ll g=primitiveroot(p),t=BSGS(g,b,p),z,z_,d=ext_gcd(a,p-1,z,z_);
    if(t%d!=0)return -1;
    ll tmp=(p-1)/d;
    z=(t/d*z%tmp+tmp)%tmp;
    return fpow(g,z,p);
}
//因数
vector<ll> factors(ll x){
    vector<ll> fac;
    ll y=x;
    for(ll i=2;i*i<=x;i++){
        if(y%i==0){
            fac.push_back(i);
            while(y%i==0)y/=i;
            if(y==1)return fac;
        }
    }
    if(y!=1)fac.push_back(y);
    return fac;
}
//欧拉函数
ll Euler(ll n){
    vector<ll> fac=factors(n);
    ll ans=n;
    for(auto p:fac)ans=ans/p*(p-1);
    return ans;
}
//莫比乌斯函数
ll Mobius(ll n){
    vector<ll> fac=factors(n);
    for(auto p:fac)n/=p;
    return n>1?0:(fac.size()&1)?-1:1;
}
int main() {
    ios::sync_with_stdio(false);
    cin.tie(0);
    cout.tie(0);
    return 0;
}
/*

杂技1. 积性函数前缀和1.1 min25筛用于求积性函数  f(x)f(x)f(x)  的前  nnn  项和，要求  fff  在素数处为一多项式，在素数的整数幂处可以直接求值。#define rep(i,a,b) for(ll i=a;i<=b;i++)
typedef long long ll;
class MultiplicativeFunction{
private:
    static const ll SQRTN=1100000;
    ll prime[SQRTN],pcnt,cnt;
    bool vis[SQRTN];
    ll d,n;
    ll val[SQRTN*2],id1[SQRTN],id2[SQRTN];
    ll G[SQRTN*2],fpval[SQRTN*2];
    ll g[SQRTN*2];
    function<ll(ll,ll)> f_prime_pow;
    ll sum_pow_n_k(ll n,ll k){
        if(k==0)return n;
        if(k==1)return n*(n+1)/2;
        if(k==2)return n*(n+1)*(2*n+1)/6;
        return -1;
    }
    void getPrime(ll n){
        pcnt=0;
        fill(vis+1,vis+1+d,0);
        rep(i,2,n){
            if(!vis[i])prime[++pcnt]=i;
            rep(j,1,pcnt){
                if(prime[j]*i>n)break;
                vis[prime[j]*i]=1;
                if(i%prime[j]==0)break;
            }
        }
    }
    ll powll(ll x,ll k){
        ll ans=1;
        while(k--)ans*=x;
        return ans;
    }
    ll &index(ll x){return x<=d?id1[x]:id2[n/x];}
    void sieve(ll x,ll k){
        rep(i,1,pcnt)fpval[i]=powll(prime[i],k)+fpval[i-1];
        rep(i,1,cnt)g[i]=sum_pow_n_k(val[i],k)-1;
        rep(i,1,pcnt){
            if(prime[i]*prime[i]>n)break;
            rep(j,1,cnt){
                if(prime[i]*prime[i]>val[j])break;
                g[j]=g[j]-powll(prime[i],k)*(g[index(val[j]/prime[i])]-fpval[i-1]);
            }
        }
    }
    ll S(ll x,ll now){
        if(x<=1 || x<prime[now])return 0;
        ll ans=G[index(x)]-fpval[now-1];
        rep(i,now,pcnt){
            if(prime[i]*prime[i]>x)break;
            ll prod=prime[i];
            for(ll k=1;prod*prime[i]<=x;k++,prod*=prime[i]){
                ans+=f_prime_pow(prime[i],k)*S(x/prod,i+1)+f_prime_pow(prime[i],k+1);
            }
        }
        return ans;
    }
public:
    ll solve(ll x,const vector<ll> &c,function<ll(ll,ll)> fp){
        f_prime_pow=fp;
        n=x;d=sqrt(n);cnt=0;
        getPrime(d);
        for(ll i=1;i<=n;i=n/(n/i)+1){
            val[++cnt]=n/i;
            index(val[cnt])=cnt;
        }
        fill(G+1,G+1+cnt,0);
        for(int k=0;k<(int)c.size();k++){
            if(c[k]!=0){
                sieve(x,k);
                rep(i,1,cnt)G[i]+=c[k]*g[i];
            }
        }
        rep(i,1,pcnt){
            fpval[i]=0;
            for(ll k=(ll)c.size()-1;k>=0;k--){
                fpval[i]=fpval[i]*prime[i]+c[k];
            }
            fpval[i]+=fpval[i-1];
        }
        return S(n,1)+1;
    }
};
使用说明：调用 solve 函数，c 是函数  fff  在素数处的解析式（多项式）的系数，fp 是  fff  在素数的整数幂处的取值计算函数。例：MultiplicativeFunction f;
//Euler函数
cout<<f.solve(n,{-1,1},[](ll p,ll k){ll a=1;while(k--)a*=p;return a/p*(p-1);})<<endl;
//Mobius函数
cout<<f.solve(n,{-1},[](ll p,ll k){return k==1?-1:0;})<<endl;
1.2 另一种筛用于求积性函数  f(x)f(x)f(x)  的前  nnn  项和，要求找一个函数  ggg ，使得  fff  与  ggg  的狄利克雷乘积、 ggg  可以快速求前  nnn  项和。class sieve{
private:
    static const ll SQRTN=100005;
    ll n,d,cnt;
    ll val[SQRTN*2],id1[SQRTN],id2[SQRTN],s[SQRTN*2];
    vector<ll> sum;
    bool vis[SQRTN*2];
    ll &index(ll x){return x<=d?id1[x]:id2[n/x];}
    function<ll(ll)> sum_g,sum_f_g;
    ll S(ll x){
        if(x<sum.size())return sum[x];
        if(x==1)return 1;
        ll p=index(x);
        if(vis[p])return val[p];
        val[p]=sum_f_g(x);
        for(ll i=2;i<=x;i=x/(x/i)+1){
            val[p]-=(sum_g(x/(x/i))-sum_g(i-1))*S(x/i);
        }
        vis[p]=1;
        return val[p];
    }
public:
    void init(function<ll(ll)> sg,function<ll(ll)> sfg,const vector<ll> &v={}){sum_g=sg;sum_f_g=sfg;sum=v;}
    ll operator () (ll x){
        n=x;
        d=sqrt(x);
        cnt=0;
        for(ll i=1;i<=x;i=x/(x/i)+1){
            val[++cnt]=x/i;
            vis[cnt]=0;
            index(val[cnt])=cnt;
        }
        return S(n);
    }
};
使用说明：调用 init 函数进行初始化，sg 为  ggg  的前  nnn  项和，sfg 为  f∗gf*gf*g  的前  nnn  项和，如果需要，可以预处理前若干项的和放到 v 中进行剪枝；直接使用 () 进行求和。例：sieve f;
//Mobius函数 mu(x) g=1 f*g=E
f.init([](ll x){return x;},[](ll x){return x>=1;});
cout<<f(n)<<endl;
//Euler函数 phi(x) g=1 f*g=id
f.init([](ll x){return x;},[](ll x){return x*(x+1)/2;});
cout<<f(n)<<endl;
//x^k phi(x) g=id^k f*g=id^(k+1)
f.init([](ll x){return x*(x+1)/2;},[](ll x){return x*(x+1)*(2*x+1)/6;});
cout<<f(n)<<endl;
//x^k mu(x) g=id^k f*g=E
f.init([](ll x){return x*(x+1)/2;},[](ll x){return x>=1;});
cout<<f(n)<<endl;
一个简化版本，使用方式相同：class sieve2{
private:
    vector<ll> sum;
    unordered_map<ll,ll> mp;
    function<ll(ll)> sum_g,sum_f_g;
    ll S(ll x){
        if(x<sum.size())return sum[x];
        if(mp.find(x)!=mp.end())return mp[x];
        ll ans=sum_f_g(x);
        for(ll i=2;i<=x;i=x/(x/i)+1){
            ans-=(sum_g(x/(x/i))-sum_g(i-1))*S(x/i);
        }
        mp[x]=ans;
        return ans;
    }
public:
    void init(function<ll(ll)> sg,function<ll(ll)> sfg,const vector<ll> &v={0,1}){
        mp.clear();
        sum_g=sg;sum_f_g=sfg;sum=v;
    }
    ll operator () (ll x){
        return S(x);
    }
};
2. 二次剩余Shanks 算法，求解二次同余方程  x2=a(modp)x^2=a\pmod px^2=a\pmod p ， ppp  是素数。ll fpow(ll a,ll b,ll c){a%=c;ll ans=1;while(b>0){if(b&1)ans=ans*a%c;b>>=1;a=a*a%c;}return ans;}
bool IsQuadraticResidue(ll a,ll p){//判断a是否为p的二次剩余 p是素数
    return fpow(a,(p-1)>>1,p)==1;
}
ll Shanks(ll a,ll p){//求解二次同余方程x^2=a(mod p) p是素数
    if(a==0)return 0;
    ll q=p-1,e=0;
    while(!(q&1))q>>=1,e++;
    static mt19937_64 rd(time(0));
    ll n=rd()%(p-1)+1;//随机选取p的一个非二次剩余，若p为定值，n也可为定值
    while(IsQuadraticResidue(n,p))n=rd()%(p-1)+1;
    ll z=fpow(n,q,p),y=z,r=e,x=fpow(a,(q-1)>>1,p),b=a*x%p*x%p;
    x=a*x%p;
    while(b!=1){
        ll temp=b*b%p,m=1;
        while(temp!=1)(temp*=temp)%=p,m++;
        if(m==r)return -1;
        ll t=y;
        rep(i,1,r-m-1)(t*=t)%=p;
        y=t*t%p,r=m%p,x=x*t%p,b=b*y%p;
    }
    return x;
}
3. 组合数在 Lucas 定理的基础上修改，可以求  CnmC_n^mC_n^m  除以  ppp  的余数， ppp  可以非素数。class Combine{
private:
    ll fpow(ll a,ll b,ll c){a%=c;ll ans=1;while(b>0){if(b&1)ans=ans*a%c;b>>=1;a=a*a%c;}return ans;}
    void factorization(ll x,vector<ll> &p,vector<ll> &w,vector<ll> &pw){
        p.clear(),w.clear(),pw.clear();
        ll y=x;
        for(ll i=2;i*i<=x;i++)if(y%i==0){
            p.push_back(i),w.push_back(0),pw.push_back(1);
            while(y%i==0)y/=i,w.back()++,pw.back()*=i;
        }
        if(y!=1)p.push_back(y),w.push_back(1),pw.push_back(y);
    }
    ll powll(ll a,ll b){ll ans=1;while(b)ans*=a,b--;return ans;}
    ll ext_gcd(ll a,ll b,ll& x,ll& y){
        ll d=a;
        if(!b)x=1,y=0;
        else d=ext_gcd(b,a%b,y,x),y-=a/b*x;
        return d;
    }
    ll inv(ll a,ll b){
        ll x,y,d=ext_gcd(a,b,x,y);
        assert(d==1);
        return (x%b+b)%b;
    }
    ll crt(const vector<ll> &m,const vector<ll> &b){
        ll ans=0,M=1;
        for(auto i:m)M*=i;
        for(int i=0;i<(int)m.size();i++){
            (ans+=M/m[i]*inv(M/m[i],m[i])%M*b[i])%=M;
        }
        return ans;
    }
public:
    ll mod;
    vector<ll> fac[105],p,w,pw,res;
    void init(ll mo){
        mod=mo;
        factorization(mod,p,w,pw);
        res.resize(p.size());
        for(int i=0;i<(int)p.size();i++){
            fac[i].resize(pw[i]);
            fac[i][0]=1;
            for(int j=1;j<pw[i];j++){
                fac[i][j]=(j%p[i]==0)?fac[i][j-1]:(fac[i][j-1]*j%pw[i]);
            }
        }
    }
    Combine(){}
    Combine(ll mod):mod(mod){init(mod);}
    ll factorial(ll n,ll i){
        if(n<p[i])return fac[i][n];
        else return fpow(fac[i][pw[i]-1],n/pw[i],pw[i])*fac[i][n%pw[i]]%pw[i]*factorial(n/p[i],i)%pw[i];
    }
    ll C_pw(ll n,ll m,ll i){
        ll c=0;
        for(ll j=p[i];j<=n;j*=p[i])c+=n/j;
        for(ll j=p[i];j<=m;j*=p[i])c-=m/j;
        for(ll j=p[i];j<=n-m;j*=p[i])c-=(n-m)/j;
        return factorial(n,i)*inv(factorial(m,i),pw[i])%pw[i]*inv(factorial(n-m,i),pw[i])%pw[i]*fpow(p[i],c,pw[i])%pw[i];
    }
    ll operator () (ll n,ll m){
        assert(m<=n && m>=0);
        for(int i=0;i<(int)p.size();i++)res[i]=C_pw(n,m,i);
        return crt(pw,res);
    }
}C;
4. 一个求和技巧（常被称为类欧几里得算法）求  ∑i=0n[ai+bc]\sum_{i=0}^n\left[\frac{ai+b}{c}\right]\sum_{i=0}^n\left[\frac{ai+b}{c}\right] ， [ ][\ ][\ ]  表示取整。ll sum_pow(ll n,ll k){
    if(k==0)return n;
    else if(k==1)return n*(n+1)/2;
    else if(k==2)return n*(n+1)*(2*n+1)/6;
    else if(k==3)return n*n*(n+1)*(n+1)/4;
    else if(k==4)return n*(2*n+1)*(n+1)*(3*n*n+3*n-1)/30;
    else assert(false);
}
ll EuclidLike1(ll a,ll b,ll c,ll n){
    if(a==0)return b/c*(n+1);
    else if(a>=c || b>=c)return (a/c)*sum_pow(n,1)+(b/c)*(n+1)+EuclidLike1(a%c,b%c,c,n);
    else return (a*n+b)/c*n-EuclidLike1(c,c-b-1,a,(a*n+b)/c-1);
}
5. 模数任意的 FFT模数任意（可取  109+710^9+710^9+7  甚至非素数）的 FFT，你可以忘掉 NTT 了。模板需要 define 相应的关键字（FFT_conv,FFT_conv_mod,FFT_conv_mod_fast）来启用相关功能。namespace FFT{
// FFT_conv:普通多项式乘积
// FFT_conv_mod:多项式乘积取模
// FFT_conv_mod_fast:优化的多项式乘积取模
#define FFT_conv_mod_fast
const int maxn=1<<20;
//typedef complex<double> cp;
struct cp{
    long double a,b;
    cp(){}
    cp(long double a,long double b):a(a),b(b){}
    cp(ll a):a(a*1.0),b(0){}
    cp operator + (const cp &t)const{return cp(a+t.a,b+t.b);}
    cp operator - (const cp &t)const{return cp(a-t.a,b-t.b);}
    cp operator * (const cp &t)const{return cp(a*t.a-b*t.b,a*t.b+b*t.a);}
    cp conj()const{return cp(a,-b);}
};
cp wn(int n,int f){
    const long double pi=acos(-1.0);
    return cp(cos(pi/n),f*sin(pi/n));
}
int g[maxn];
void DFT(cp *a,int n,int f){
    for(int i=0;i<n;i++)if(i>g[i])swap(a[i],a[g[i]]);
    for(int i=1;i<n;i<<=1){
        cp w=wn(i,f),x,y;
        for(int j=0;j<n;j+=i+i){
            cp e(1,0);
            for(int k=0;k<i;e=e*w,k++){
                x=a[j+k];
                y=a[j+k+i]*e;
                a[j+k]=x+y;
                a[j+k+i]=x-y;
            }
        }
    }
    if(f==-1){
        cp Inv(1.0/n,0);
        for(int i=0;i<n;i++)a[i]=a[i]*Inv;
    }
}

#ifdef FFT_conv
cp a[maxn],b[maxn],c[maxn];
vector<ll> conv(const vector<ll> &u,const vector<ll> &v,ll sz){
    int k=0,s=1,n=(int)u.size()-1,m=(int)v.size()-1;
    while(s<n+m+1)k++,s<<=1;
    for(int i=1;i<s;i++)g[i]=(g[i/2]/2)|((i&1)<<(k-1));
    for(int i=0;i<s;i++)a[i]=i<=n?u[i]:0;
    for(int i=0;i<s;i++)b[i]=i<=m?v[i]:0;
    DFT(a,s,1);
    DFT(b,s,1);
    for(int i=0;i<s;i++)c[i]=a[i]*b[i];
    DFT(c,s,-1);
    vector<ll> ans(sz+1,0);
    for(int i=0;i<=sz;i++)ans[i]=i<s?ll(round(c[i].a)):0;
    return ans;
}
#endif

#ifdef FFT_conv_mod
cp a1[maxn],a2[maxn],b1[maxn],b2[maxn],c[maxn];
vector<ll> conv_mod(const vector<ll> &u,const vector<ll> &v,int sz,ll mod){
    int k=0,s=1,n=(int)u.size()-1,m=(int)v.size()-1,M=sqrt(mod)+1;
    while(s<n+m+1)k++,s<<=1;
    for(int i=1;i<s;i++)g[i]=(g[i/2]/2)|((i&1)<<(k-1));
    for(int i=0;i<s;i++){
        if(i<=n)a1[i]=u[i]%mod/M,b1[i]=u[i]%mod%M;
        else a1[i]=b1[i]=0;
        if(i<=m)a2[i]=v[i]%mod/M,b2[i]=v[i]%mod%M;
        else a2[i]=b2[i]=0;
    }
    DFT(a1,s,1);
    DFT(b1,s,1);
    DFT(a2,s,1);
    DFT(b2,s,1);
    vector<ll> ans(sz+1,0);
    for(int i=0;i<s;i++)c[i]=a1[i]*a2[i];
    DFT(c,s,-1);
    for(int i=0;i<min(sz+1,s);i++)(ans[i]+=ll(round(c[i].a))%mod*M%mod*M)%=mod;
    for(int i=0;i<s;i++)c[i]=a1[i]*b2[i]+a2[i]*b1[i];
    DFT(c,s,-1);
    for(int i=0;i<min(sz+1,s);i++)(ans[i]+=ll(round(c[i].a))%mod*M)%=mod;
    for(int i=0;i<s;i++)c[i]=b1[i]*b2[i];
    DFT(c,s,-1);
    for(int i=0;i<min(sz+1,s);i++)(ans[i]+=ll(round(c[i].a))%mod)%=mod;
    return ans;
}
#endif

#ifdef FFT_conv_mod_fast
cp a[maxn],b[maxn],Aa[maxn],Ab[maxn],Ba[maxn],Bb[maxn];
vector<ll> conv_mod_fast(const vector<ll> &u,const vector<ll> &v,int sz,ll mod){
    int k=0,s=1,n=(int)u.size()-1,m=(int)v.size()-1,M=sqrt(mod)+1;
    while(s<n+m+1)k++,s<<=1;
    for(int i=1;i<s;i++)g[i]=(g[i/2]/2)|((i&1)<<(k-1));
    for(int i=0;i<s;i++){
        if(i<=n)a[i].a=u[i]%mod%M,a[i].b=u[i]%mod/M;
        else a[i]=0;
        if(i<=m)b[i].a=v[i]%mod%M,b[i].b=v[i]%mod/M;
        else b[i]=0;
    }
    DFT(a,s,1);
    DFT(b,s,1);
    for(int i=0;i<s;i++){
        int j=(s-i)%s;
        cp t1=(a[i]+a[j].conj())*cp(0.5,0);
        cp t2=(a[i]-a[j].conj())*cp(0,-0.5);
        cp t3=(b[i]+b[j].conj())*cp(0.5,0);
        cp t4=(b[i]-b[j].conj())*cp(0,-0.5);
        Aa[i]=t1*t3,Ab[i]=t1*t4,Ba[i]=t2*t3,Bb[i]=t2*t4;
    }
    for(int i=0;i<s;i++){
        a[i]=Aa[i]+Ab[i]*cp(0,1);
        b[i]=Ba[i]+Bb[i]*cp(0,1);
    }
    DFT(a,s,-1);
    DFT(b,s,-1);
    vector<ll> ans(sz+1,0);
    for(int i=0;i<min(s,sz+1);i++){
        ll t1=(ll)round(a[i].a)%mod;
        ll t2=(ll)round(a[i].b)%mod;
        ll t3=(ll)round(b[i].a)%mod;
        ll t4=(ll)round(b[i].b)%mod;
        (ans[i]+=t1+(t2+t3)*M%mod+t4*M*M)%=mod;
    }
    return ans;
}
#endif
}
6. 多项式插值6.1 Lagrange 插值#define rep(i,a,b) for(ll i=a;i<=b;i++)
const ll mo=1e9+7;
ll fpow(ll a,ll b){ll ans=1;while(b>0){if(b&1)ans=ans*a%mo;b>>=1;a=a*a%mo;}return ans;}
ll lagrange(vector<ll> x,vector<ll> y,ll X){
    auto p=y.begin();
    ll ans=0;
    for(auto k:x){
        ll a=*p++%mo,b=1;
        for(auto j:x)if(j!=k)a=(X-j)%mo*a%mo,b=(k-j)%mo*b%mo;
        ans=(ans+mo+a*fpow(b,mo-2)%mo)%mo;
    }
    return ans;
}
插值点连续的情况下， xi=i,i=1,2,⋯,nx_i=i,i=1,2,\cdots,nx_i=i,i=1,2,\cdots,n ，时间复杂度优化到  O(n)O(n)O(n) ， nnn  是插值点的个数。#define rep(i,a,b) for(ll i=a;i<=b;i++)
ll Lagrange(const vector<ll> &y,ll x,ll mod){
    ll n=y.size()-1;
    static const ll N=100005;
    static ll lx[N],rx[N],inv[N],mul[N];
    rep(i,1,n)inv[i]=(i==1)?1:((mod-mod/i)*inv[mod%i]%mod);
    rep(i,0,n)lx[i]=rx[i]=(x-1-i)%mod;
    rep(i,1,n)(lx[i]*=lx[i-1])%=mod;
    for(ll i=n-1;i>=0;i--)(rx[i]*=rx[i+1])%=mod;
    rep(i,0,n)mul[i]=(i==0)?1:(mul[i-1]*inv[i])%mod;
    ll ans=0;
    rep(i,0,n){
        ll now=1;
        if(i>0)(now*=lx[i-1]*mul[i]%mod)%=mod;
        if(i<n)(now*=rx[i+1]*mul[n-i]%mod)%=mod;
        if((n-i)&1)now=-now;
        (ans+=y[i]*now)%=mod;
    }
    return ans+(ans<0)*mod;
}
二维的 Lagrange 插值#define rep(i,a,b) for(ll i=a;i<=b;i++)
type lagrange2(vector<type> x,vector<type> y,vector<vector<type> > z,type X,type Y){
    int M=x.size()-1,N=y.size()-1;
    type ans=0;
    rep(m,0,M)rep(n,0,N){
        type t=z[m][n];
        rep(i,0,M)if(i!=m)t*=(X-x[i])/(x[m]-x[i]);
        rep(i,0,N)if(i!=n)t*=(Y-y[i])/(y[n]-y[i]);
        ans+=t;
    }
    return ans;
}
6.2 牛顿插值#define rep(i,a,b) for(ll i=a;i<=b;i++)
ll fpow(ll a,ll b){ll ans=1;while(b>0){if(b&1)ans=ans*a%mo;b>>=1;a=a*a%mo;}return ans;}
class NewtonPoly{
    public:
    ll f[105],d[105],x[105],n=0;
    void add(ll X,ll Y){
        x[n]=X,f[n]=Y%mo;
        rep(i,1,n)f[n-i]=(f[n-i+1]-f[n-i])%mo*fpow((x[n]-x[n-i])%mo,mo-2)%mo;
        d[n++]=f[0];
    }
    ll cal(ll X){
        ll ans=0,t=1;
        rep(i,0,n-1)ans=(ans+d[i]*t)%mo,t=(X-x[i])%mo*t%mo;
        return ans+mo*(ans<0);
    }
};
6.3 利用插值求nkn^kn^k  前  nnn  项和#define rep(i,a,b) for(ll i=a;i<=b;i++)
const ll mo=1e9+7;
ll sum_pow_n_k(ll n,ll k){
    static ll inv[1005],d[1005][1005],init_flag=1;
    if(init_flag){
        init_flag=0;
        const ll all=1002;
        inv[1]=1;
        rep(i,2,all)inv[i]=(mo-mo/i)*inv[mo%i]%mo;
        d[1][1]=1;
        rep(i,2,all)rep(j,1,i)d[i][j]=(j*j*d[i-1][j]+(j-1)*d[i-1][j-1])%mo*inv[j]%mo;
    }
    n%=mo;
    ll ans=0,now=n;
    rep(i,1,k+1){
        (ans+=d[k+1][i]*now)%=mo;
        (now*=(n-i))%=mo;
    }
    return ans+(ans<0)*mo;
}
7. 线性递推数列7.1 BM 算法根据数列前若干项求递推公式。例如斐波那契数列，传入 {1,1,2,3,5,8,13}，传出 {1,-1,-1}（ Fn−Fn−1−Fn−2=0F_n-F_{n-1}-F_{n-2}=0F_n-F_{n-1}-F_{n-2}=0 ）。#define rep(i,a,b) for(ll i=a;i<=b;i++)
const ll N=25;
const ll mo=1e9+7;
ll fpow(ll a,ll b){ll ans=1;while(b>0){if(b&1)ans=ans*a%mo;b>>=1;a=a*a%mo;}return ans;}
//BM算法 求线性递推数列的递推公式
vector<ll> BM(const vector<ll> &s){
    vector<ll> C={1},B={1},T;
    ll L=0,m=1,b=1;
    rep(n,0,s.size()-1){
        ll d=0;
        rep(i,0,L)d=(d+s[n-i]%mo*C[i])%mo;
        if(d==0)m++;
        else{
            T=C;
            ll t=mo-fpow(b,mo-2)*d%mo;
            while(C.size()<B.size()+m)C.push_back(0);
            rep(i,0,B.size()-1)C[i+m]=(C[i+m]+t*B[i])%mo;
            if(2*L>n)m++;
            else L=n+1-L,B=T,b=d,m=1;
        }
    }
    return C;
}
7.2 线性递推数列求单项值利用多项式求  FnF_nF_n ，复杂度为  O(k2log⁡n)O(k^2\log n)O(k^2\log n) ， kkk  为递推公式长度。s 为前若干项，C 为递推公式系数， nnn  从  000  开始。#define rep(i,a,b) for(ll i=a;i<=b;i++)
ll polySolve(const vector<ll> &s,const vector<ll> &C,ll n){
    if(n<s.size())return s[n];
    static ll g[N],f[N],d[N];
    ll k=(ll)C.size()-1,w=1;
    rep(i,0,k)f[i]=i==1,d[i]=i==k?1:C[k-i];
    while((w<<1)<=n)w<<=1;
    while(w>>=1){
        rep(i,0,k+k-2)g[i]=0;
        rep(i,0,k-1)if(f[i])rep(j,0,k-1)(g[i+j]+=f[i]*f[j])%=mo;
        for(ll i=k+k-2;i>=k;i--)if(g[i])rep(j,1,k)(g[i-j]-=g[i]*d[k-j])%=mo;
        rep(i,0,k-1)f[i]=g[i];
        if(w&n)for(ll i=k;i>=0;i--)f[i]=i==k?f[i-1]:(i==0?-f[k]*d[i]:(f[i-1]-f[k]*d[i]))%mo;
    }
    ll ans=0;
    rep(i,0,k-1)(ans+=f[i]*s[i])%=mo;
    return ans+(ans<0)*mo;
}
7.3 RS 算法当模数非素数时，使用 RS 算法替代 BM 算法。模板是网上搜刮来的，它终结了线性递推题目在ACM中的历史，模板作者留了一个 FFT 的优化接口，供下一个毒瘤出题人使用。struct LinearSequence {
    typedef vector<long long> vec;

    static void extand(vec &a, size_t d, ll value = 0) {
        if (d <= a.size()) return;
        a.resize(d, value);
    }

    static vec BerlekampMassey(const vec &s, ll mod) {
        function<ll(ll)> inverse = [&](ll a) {
            return a == 1 ? 1 : (ll) (mod - mod / a) * inverse(mod % a) % mod;
        };
        vec A = {1}, B = {1};
        ll b = s[0];
        for (size_t i = 1, m = 1; i < s.size(); ++i, m++) {
            ll d = 0;
            for (size_t j = 0; j < A.size(); ++j) {
                d += A[j] * s[i - j] % mod;
            }
            if (!(d %= mod)) continue;
            if (2 * (A.size() - 1) <= i) {
                auto temp = A;
                extand(A, B.size() + m);
                ll coef = d * inverse(b) % mod;
                for (size_t j = 0; j < B.size(); ++j) {
                    A[j + m] -= coef * B[j] % mod;
                    if (A[j + m] < 0) A[j + m] += mod;
                }
                B = temp, b = d, m = 0;
            } else {
                extand(A, B.size() + m);
                ll coef = d * inverse(b) % mod;
                for (size_t j = 0; j < B.size(); ++j) {
                    A[j + m] -= coef * B[j] % mod;
                    if (A[j + m] < 0) A[j + m] += mod;
                }
            }
        }
        return A;
    }

    static void exgcd(ll a, ll b, ll &g, ll &x, ll &y) {
        if (!b) x = 1, y = 0, g = a;
        else {
            exgcd(b, a % b, g, y, x);
            y -= x * (a / b);
        }
    }

    static ll crt(const vec &c, const vec &m) {
        int n = c.size();
        ll M = 1, ans = 0;
        for (int i = 0; i < n; ++i) M *= m[i];
        for (int i = 0; i < n; ++i) {
            ll x, y, g, tm = M / m[i];
            exgcd(tm, m[i], g, x, y);
            ans = (ans + tm * x * c[i] % M) % M;
        }
        return (ans + M) % M;
    }

    static vec ReedsSloane(const vec &s, ll mod) {
        auto inverse = [](ll a, ll m) {
            ll d, x, y;
            exgcd(a, m, d, x, y);
            return d == 1 ? (x % m + m) % m : -1;
        };
        auto L = [](const vec & a, const vec & b) {
            int da = (a.size() > 1 || (a.size() == 1 && a[0])) ? a.size() - 1 : -1000;
            int db = (b.size() > 1 || (b.size() == 1 && b[0])) ? b.size() - 1 : -1000;
            return max(da, db + 1);
        };
        auto prime_power = [&](const vec & s, ll mod, ll p, ll e) {
            // linear feedback shift register mod p^e, p is prime
            vector<vec> a(e), b(e), an(e), bn(e), ao(e), bo(e);
            vec t(e), u(e), r(e), to(e, 1), uo(e), pw(e + 1);;
            pw[0] = 1;
            for (int i = pw[0] = 1; i <= e; ++i) pw[i] = pw[i - 1] * p;
            for (ll i = 0; i < e; ++i) {
                a[i] = {pw[i]}, an[i] = {pw[i]};
                b[i] = {0}, bn[i] = {s[0] * pw[i] % mod};
                t[i] = s[0] * pw[i] % mod;
                if (t[i] == 0) {
                    t[i] = 1, u[i] = e;
                } else {
                    for (u[i] = 0; t[i] % p == 0; t[i] /= p, ++u[i]);
                }
            }
            for (size_t k = 1; k < s.size(); ++k) {
                for (int g = 0; g < e; ++g) {
                    if (L(an[g], bn[g]) > L(a[g], b[g])) {
                        ao[g] = a[e - 1 - u[g]];
                        bo[g] = b[e - 1 - u[g]];
                        to[g] = t[e - 1 - u[g]];
                        uo[g] = u[e - 1 - u[g]];
                        r[g] = k - 1;
                    }
                }
                a = an, b = bn;
                for (int o = 0; o < e; ++o) {
                    ll d = 0;
                    for (size_t i = 0; i < a[o].size() && i <= k; ++i) {
                        d = (d + a[o][i] * s[k - i]) % mod;
                    }
                    if (d == 0) {
                        t[o] = 1, u[o] = e;
                    } else {
                        for (u[o] = 0, t[o] = d; t[o] % p == 0; t[o] /= p, ++u[o]);
                        int g = e - 1 - u[o];
                        if (L(a[g], b[g]) == 0) {
                            extand(bn[o], k + 1);
                            bn[o][k] = (bn[o][k] + d) % mod;
                        } else {
                            ll coef = t[o] * inverse(to[g], mod) % mod * pw[u[o] - uo[g]] % mod;
                            int m = k - r[g];
                            extand(an[o], ao[g].size() + m);
                            extand(bn[o], bo[g].size() + m);
                            for (size_t i = 0; i < ao[g].size(); ++i) {
                                an[o][i + m] -= coef * ao[g][i] % mod;
                                if (an[o][i + m] < 0) an[o][i + m] += mod;
                            }
                            while (an[o].size() && an[o].back() == 0) an[o].pop_back();
                            for (size_t i = 0; i < bo[g].size(); ++i) {
                                bn[o][i + m] -= coef * bo[g][i] % mod;
                                if (bn[o][i + m] < 0) bn[o][i + m] -= mod;
                            }
                            while (bn[o].size() && bn[o].back() == 0) bn[o].pop_back();
                        }
                    }
                }
            }
            return make_pair(an[0], bn[0]);
        };

        vector<tuple<ll, ll, int>> fac;
        for (ll i = 2; i * i <= mod; ++i)
            if (mod % i == 0) {
                ll cnt = 0, pw = 1;
                while (mod % i == 0) mod /= i, ++cnt, pw *= i;
                fac.emplace_back(pw, i, cnt);
            }
        if (mod > 1) fac.emplace_back(mod, mod, 1);
        vector<vec> as;
        size_t n = 0;
        for (auto && x : fac) {
            ll mod, p, e;
            vec a, b;
            tie(mod, p, e) = x;
            auto ss = s;
            for (auto && x : ss) x %= mod;
            tie(a, b) = prime_power(ss, mod, p, e);
            as.emplace_back(a);
            n = max(n, a.size());
        }
        vec a(n), c(as.size()), m(as.size());
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < as.size(); ++j) {
                m[j] = get<0>(fac[j]);
                c[j] = i < as[j].size() ? as[j][i] : 0;
            }
            a[i] = crt(c, m);
        }
        return a;
    }

    LinearSequence(const vec &s, const vec &c, ll mod) :
        init(s), trans(c), mod(mod), m(s.size()) {}

    LinearSequence(const vec &s, ll mod, bool is_prime = true) : mod(mod) {
        vec A;
        if (is_prime) A = BerlekampMassey(s, mod);
        else A = ReedsSloane(s, mod);
        if (A.empty()) A = {0};
        m = A.size() - 1;
        trans.resize(m);
        for (int i = 0; i < m; ++i) {
            trans[i] = (mod - A[i + 1]) % mod;
        }
        reverse(trans.begin(), trans.end());
        init = {s.begin(), s.begin() + m};
    }

    ll calc(ll n) {
        if (mod == 1) return 0;
        if (n < m) return init[n];
        vec v(m), u(m << 1);
        int msk = !!n;
        for (ll m = n; m > 1; m >>= 1) msk <<= 1;
        v[0] = 1 % mod;
        for (int x = 0; msk; msk >>= 1, x <<= 1) {
            fill_n(u.begin(), m * 2, 0);
            x |= !!(n & msk);
            if (x < m) u[x] = 1 % mod;
            else {// can be optimized by fft/ntt
                for (int i = 0; i < m; ++i) {
                    for (int j = 0, t = i + (x & 1); j < m; ++j, ++t) {
                        u[t] = (u[t] + v[i] * v[j]) % mod;
                    }
                }
                for (int i = m * 2 - 1; i >= m; --i) {
                    for (int j = 0, t = i - m; j < m; ++j, ++t) {
                        u[t] = (u[t] + trans[j] * u[i]) % mod;
                    }
                }
            }
            v = {u.begin(), u.begin() + m};
        }
        ll ret = 0;
        for (int i = 0; i < m; ++i) {
            ret = (ret + v[i] * init[i]) % mod;
        }
        return ret;
    }

    vec init, trans;
    ll mod;
    int m;
};
8. 数值积分8.1 复化 Simpson 公式#define rep(i,a,b) for(ll i=a;i<=b;i++)
//复化Simpson公式 O(n)
template<typename type,typename fun> type Simpson(fun f,type a,type b,int n){
    type h=(b-a)/n,s=0;
    rep(i,0,n-1)s+=h/6.0*(f(a+h*i)+4.0*f(a+h*(i+0.5))+f(a+h*(i+1.0)));
    return s;
}
8.2 Romberg 算法#define rep(i,a,b) for(ll i=a;i<=b;i++)
//Romberg算法 O(2^n)
template<typename type,typename fun> type Romberg(fun f,type a,type b,int n){
    assert(n>=3);
    type h=b-a,T[n+2],S[n+2],C[n+2],R[n+2];
    T[1]=h/2.0*(f(a)+f(b));
    rep(t,0,n-1){
        type s=0;
        rep(i,0,(1<<t)-1)s+=f(a+h*(i+0.5));
        T[t+2]=T[t+1]/2.0+h*s/2.0;
        h/=2.0;
    }
    rep(i,1,n)S[i]=T[i+1]*4.0/3.0-T[i]/3.0;
    rep(i,1,n-1)C[i]=S[i+1]*16.0/15.0-S[i]/15.0;
    rep(i,1,n-2)R[i]=C[i+1]*64.0/63.0-C[i]/63.0;
    return R[n-2];
}
 */