#include <bits/stdc++.h>

using namespace std;
typedef long long ll;

struct DSU {
    std::vector<int> f, siz;
    
    DSU() {}
    DSU(int n) {
        init(n);
    }
    
    void init(int n) {
        f.resize(n);
        std::iota(f.begin(), f.end(), 0);
        siz.assign(n, 1);
    }
    
    int find(int x) {
        while (x != f[x]) {
            x = f[x] = f[f[x]];
        }
        return x;
    }
    
    bool same(int x, int y) {
        return find(x) == find(y);
    }
    
    bool merge(int x, int y) {
        x = find(x);
        y = find(y);
        if (x == y) {
            return false;
        }
        siz[x] += siz[y];
        f[y] = x;
        return true;
    }
    
    int size(int x) {
        return siz[find(x)];
    }
};

struct DSU2 {//带权并查集
    vector<int> fa, siz;
    vector<ll> dis;
    DSU2(int n) : fa(n + 1), siz(n + 1, 1), dis(n + 1) { iota(fa.begin(), fa.end(), 0); };
    function<int(int)> find = [&](int x) {
        if (x == fa[x]) return x;
        find(fa[x]); dis[x] += dis[fa[x]];
        return fa[x] = fa[fa[x]];
    };
    int size(int x) { return siz[find(x)]; }
    ll dist(int x, int y) {
        int xf = find(x), yf = find(y);
        if (xf != yf) return -1;
        return dis[x] + -dis[y];
    }
    bool same(int x, int y) { return find(x) == find(y); }
    bool merge(int x, int y, ll w) {
        int xf = find(x), yf = find(y);
        if (xf == yf) return false;
        siz[yf] += siz[xf]; fa[xf] = yf;
        dis[xf] = -dis[x] + w + dis[y];
        return true;
    }
};

int main() {

}