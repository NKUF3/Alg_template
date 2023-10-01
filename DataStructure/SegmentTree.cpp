#include <bits/stdc++.h>

using namespace std;
typedef long ll;

template<class Info>
struct Segtree {
#define lson k << 1, l, mid
#define rson k << 1 | 1, mid + 1, r
    int n;
    vector<Info> info;
    Segtree(int _n) : n(_n), info((_n + 5) << 2) {};
    Segtree(vector<Info>& arr) : Segtree(arr.size() - 1) {
        function<void(int, int, int)> build = [&](int k, int l, int r) {
            if (l == r) {
                info[k] = arr[l];
                return;
            }
            int mid = l + r >> 1;
            build(lson), build(rson);
            pushup(k);
        };
        build(1, 1, n);
    }
    void pushup(int k) {
        info[k] = merge(info[k << 1], info[k << 1 | 1]);
    }
    void upd(int k, int l, int r, int x, Info& z) {
        if (l == r) return void(info[k] = z);
        int mid = l + r >> 1;
        if (x <= mid) upd(lson, x, z);
        else upd(rson, x, z);
        pushup(k);
    }
    Info qry(int k, int l, int r, int x, int y) {
        if (l > y || r < x) return Info();
        if (l >= x && r <= y) return info[k];
        int mid = l + r >> 1;
        return merge(qry(lson, x, y), qry(rson, x, y));
    }
    void upd(int pos, Info& z) {
        upd(1, 1, n, pos, z);
    }
    Info qry(int l, int r) {
        return qry(1, 1, n, l, r);
    }
};

struct Info {
    ll val;
    friend Info merge(const Info& a, const Info& b) {
        Info res;
        res.val = max(a.val, b.val);
        return res;
    }
};

// ==================================================================================
// =================================Lazy Tag=========================================
// ==================================================================================

template<class Info, class Tag>
struct Segtree {
#define lson k << 1, l, mid
#define rson k << 1 | 1, mid + 1, r
    int n;
    vector<Info> info;
    vector<Tag> tag;
    Segtree(int _n) : n(_n), info((_n + 5) << 2), tag((_n + 5) << 2) {};
    Segtree(vector<Info>& arr) : Segtree(arr.size() - 1) {
        function<void(int, int, int)> build = [&](int k, int l, int r) {
            if (l == r) {
                info[k] = arr[l];
                return;
            }
            int mid = l + r >> 1;
            build(lson), build(rson);
            pushup(k);
        };
        build(1, 1, n);
    }
    void pushdown(int k, int l, int r) {
        int mid = l + r >> 1, lt = k << 1, rt = k << 1 | 1;
        info[lt].down(tag[k], mid - l + 1);
        info[rt].down(tag[k], r - mid);
        tag[lt].down(tag[k]);
        tag[rt].down(tag[k]);
        tag[k] = Tag();//初始化tag
    }
    void pushup(int k) {
        info[k] = merge(info[k << 1], info[k << 1 | 1]);
    }
    void sigupd(int k, int l, int r, int x, int y, Info& z) {
        if (l > y || r < x) return;
        if (l == r) return void(info[k] = z);
        if (tag[k].change()) pushdown(k, l, r);
        int mid = l + r >> 1;
        sigupd(lson, x, y, z), sigupd(rson, x, y, z);
        pushup(k);
    }
    void upd(int k, int l, int r, int x, int y, Tag& z) {
        if (l > y || r < x) return;
        if (l >= x && r <= y) {
            info[k].down(z, r - l + 1);
            tag[k].down(z);
            return;
        }
        if (tag[k].change()) pushdown(k, l, r);
        int mid = l + r >> 1;
        upd(lson, x, y, z), upd(rson, x, y, z);
        pushup(k);
    }
    Info qry(int k, int l, int r, int x, int y) {
        if (l > y || r < x) return Info();
        if (l >= x && r <= y) return info[k];
        if (tag[k].change()) pushdown(k, l, r);
        int mid = l + r >> 1;
        return merge(qry(lson, x, y), qry(rson, x, y));
    }
    void sigupd(int x, int y, Info& z) {
        sigupd(1, 1, n, x, y, z);
    }
    void upd(int l, int r, Tag& z) {
        upd(1, 1, n, l, r, z);
    }
    Info qry(int l, int r) {
        return qry(1, 1, n, l, r);
    }
};

struct Tag {
    int val;
    bool change() { return val != 0; }
    void down(const Tag& t) {
        //标记下放
        val += t.val;
    }
};

struct Info {
    ll sum;
    void down(const Tag& t, int len) {
        //标记下放
        sum += t.val * len;
    }
    friend Info merge(const Info& a, const Info& b) {
        Info res;
        res.sum = a.sum + b.sum;
        return res;
    }
};