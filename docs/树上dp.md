---
title: acm dp学习
date: 2023-11-01 16:21:41
tags: [算法]
mathjax: true
---

<center>动态规划</center>

<!--more-->

# 树形动态规划

## 树上背包

### P1
给定一个n个点的有根树，每个点有一个权值。要求对于每个点，回答在它的子树中选择一个大小恰好为m的包含这个点的连通块的最大权值和。

考虑答案的结构，以u为根的子树中选取大小为m的最优的连通块方案中，u的孩子的选取方案也一定是最优的。我们要解决的便是把m-1个点分配个u的所有孩子$c_1,c_2...c_n$，每个孩子对应的子树选择$m_1,m_2...m_n$个点，求最大可能的权值和。这显然是一个背包问题。

```cpp
// O(nm) 在某个子树内包含子树根的m个点的连通块的最大权值和
#include <bits/stdc++.h>
using namespace std;

const int MAXN = 2005;
const int MAXM = MAXN;
const int inf = 1 << 29;
vector<int> tree[MAXN];
int dp[MAXN][MAXM];
int sz[MAXN];
int n, q, M = 100;

void dfs(int u) {
    sz[u] = 1;
    static int tmp[MAXN];
    for (auto son : tree[u]) {
        dfs(son);
        for (int i = 1; i <= min(M, sz[u] + sz[son]); i++)
            tmp[i] = -inf;
        for (int i = 1; i <= min(M, sz[u]); i++) {
            for (int j = 0; j <= min(M - i, sz[son]); j++) {
                tmp[i + j] = max(tmp[i + j], dp[u][i] + dp[son][j]);
            }
        }
        for (int i = 1; i <= min(M, sz[u] + sz[son]); i++)
            dp[u][i] = tmp[i];
        sz[u] += sz[son];
    }
}

int main() {
    cin >> n >> q;
    for (int i = 2; i <= n; i++) {
        int f;
        cin >> f;
        tree[f].push_back(i);
    }
    for (int i = 1; i <= n; i++)
        cin >> dp[i][1];
    dfs(1);
    while (q--) {
        int u, m;
        cin >> u >> m;
        cout << dp[u][m] << endl;
    }
    return 0;
}
```

### P2
给定一个n个点的有根树，每个点有一个重量$w_i$和权值$v_i$。要求回答选择一个重量和**恰好**为m的**包含根的连通块**最大的权值和。

对于该问题，我们可以使用和上一题类似的解法来解，令$dp[i][j]$表示i这棵子树选择重量和恰好为j的连通块的最大权值和。但是，对于每个节点，我们都需要$O(m^2)$的时间复杂度去更新dp数组，最终的时间复杂度为$O(nm^2)$，比较耗时。

这里我们可以使用一个技巧，即在**dfs序**上做dp，将问题转化为一个特殊的线性问题。如下图所示的树，对应的dfs序为：$3\ 1\ 4\ 8\ 6\ 2\ 5\ 7$。这里我们定义r函数，$r(u)$表示dfs序中在$u$这棵子树后被遍历的第一个节点在dfs序中的下标。比如下图中1对应的子树遍历完后首先将会遍历节点2，则$r(1)=6$，即2在dfs序中的下标(这里下标从1开始计数)。
![tree](https://raw.githubusercontent.com/yuwensq/imgBase/master/202311011652728.jpg)

根据题意我们可以得知，如果一个节点不选，那么该节点对应的子树中其余的节点也一定不会被选(因为已经和根节点不连通了)，也就是直接跳过这棵子树。这里我们定义$dp[i][j]$表示在dfs序下标从i到n的节点中选取重量和恰为j的若干节点的最大权值，且满足若某一个节点被选择，其在dfs序中位于i到n的祖先也一定被选择。则我们可以得到状态转移方程：$dp[i][j]=max(dp[r[dfs[i]]][j],dp[i+1][j-w_i]+v_i)$，其中$dfs[i]$表示dfs序中第i个节点的编号。状态转移含义为，如果下标为i的节点不选，那么它对应的子树也不选，直接忽略整颗子树，从$r[dfs[i]]$继续考虑即可；如果选择$dfs[i]$，那么可以继续考虑其孩子是否选择，即继续考虑下标为i+1的节点。

最终时间复杂度为$O(nm)$。

```cpp
// O(nm) dfs序
#include <bits/stdc++.h>
using namespace std;

const int N = 2005;
const int inf = 1 << 29;
vector<int> tree[N];
int dp[N][N];
int w[N], a[N], l[N], id[N], r[N];

int n, m, tot;

void dfs(int u) {
    l[u] = ++tot;
    id[tot] = u;
    for (auto v : tree[u])
        dfs(v);
    r[u] = tot;
}

int main() {
    cin >> n >> m;
    for (int i = 2; i <= n; i++) {
        int f;
        cin >> f;
        tree[f].push_back(i);
    }
    for (int i = 1; i <= n; i++)
        cin >> a[i];
    for (int i = 1; i <= n; i++)
        cin >> w[i];
    dfs(1);
    for (int j = 1; j <= m; j++)
        dp[n + 1][j] = -inf;
    for (int i = n; i >= 1; i--) {
        int u = id[i];
        for (int j = 0; j <= m; j++) {
            dp[i][j] = dp[r[u] + 1][j]; // 不选u, u对应的子树都不能选
            if (j >= w[u])
                dp[i][j] = max(dp[i][j], dp[i + 1][j - w[u]] + a[u]);
        }
    }
    for (int i = 0; i <= m; i++)
        if (dp[1][i] >= 0)
            cout << dp[1][i] << endl;
        else
            cout << 0 << endl;
    return 0;
}
```

## 树上路径

### P1
给定一个n个节点的树和m条树上的简单路径，每个路径有一个权值，求不相交路径的最大权值和。

令$dp[u]$表示u这棵子树中不超出u点的不相交路径的最大权值和。这里的不超出u点指的是选择的路径不能从u点继续向上延申，也就是说路径两个端点的lca最高能到达u。比如下图中，$dp[1]$中不能包含选择路径$3148$的情况。![tree](https://raw.githubusercontent.com/yuwensq/imgBase/master/202311011747340.jpg)

接下来我们考虑状态转移，对于u这棵子树，如果不选择任何通过u节点的路径，那么$dp[u]= \sum_{i \in childs[u]} dp[i]$。如果选择了一条通过u节点的路径，$dp[u]$为从子树u中删除该路径后得到的森林中所有树的根节点的dp值的和与该路径的权值的和。以上图为例，如果我们选择了$3148$这条路径，那么$dp[3]=max(dp[3], dp[6]+dp[2]+v[3148])$。但是上述过程编程比较麻烦，这里进一步观察我们可以发现，如果不选择通过节点3的路径，$dp[3]_1=dp[1]+dp[2]$。我们先不考虑路径权值(最后加上就行)，如果选择路径$31$，$dp[3]_2=dp[4]+dp[6]+dp[2]=dp[3]_1+(\sum_{i \in childs[1]} dp[i]-dp[1])$。更进一步，如果选择路径$314$，$dp[3]_3=dp[8]+dp[6]+dp[2]=dp[3]_2+(\sum_{i \in childs[4]} dp[i]-dp[4])$。我们可以发现，我们每删除一个节点x，dp的增量为$\delta _x=\sum_{i \in childs[x]} dp[i]-dp[x]$。而$dp[u]$的初始值为$\sum_{i \in childs[u]} dp[i]$，这样我们只需要遍历一遍路径上所有的点便可以计算出选择该通过u的路径的情况下$dp[u]$的值。

```cpp
// O(nm) 每个路径一个权值，问不相交的路径的最大权值和
#include <bits/stdc++.h>
using namespace std;

typedef long long ll;

const int N = 2010;
const ll inf = 1ll << 60;

int n, m;
vector<int> son[N];
vector<array<int, 3>> path[N];
int father[N];
int dep[N];

ll dp[N], sdp[N];

void dfs(int u) {
    for (auto v : son[u]) {
        dfs(v);
        sdp[u] += dp[v];
    }
    dp[u] = sdp[u];
    for (auto p : path[u]) {
        ll tmp = sdp[u];
        int x = p[0];
        while (x != u) {
            tmp += sdp[x] - dp[x];
            x = father[x];
        }
        x = p[1];
        while (x != u) {
            tmp += sdp[x] - dp[x];
            x = father[x];
        }
        tmp += p[2];
        dp[u] = max(dp[u], tmp);
    }
}

int main() {
    cin >> n >> m;
    for (int i = 2; i <= n; i++) {
        cin >> father[i];
        son[father[i]].push_back(i);
        dep[i] = dep[father[i]] + 1; // 因为输入是按照顺序的，这里这样也行
    }
    for (int i = 1; i <= m; i++) {
        int u, v, a;
        cin >> u >> v >> a;
        int x = u, y = v;
        while (x != y) {
            if (dep[x] > dep[y]) x = father[x];
            else y = father[y];
        }
        path[x].push_back({u, v, a});
    }
    dfs(1);
    cout << dp[1];
    return 0;
}
```

### P2
给定一个n个点的有根树。给出m条树上的简单路径，每个路径有一个权值。保证每个路径都是从一个点到它的祖先。要求选择一些路径，使得每个点至少在一条路径上，并且路径的权值和最小。

令$dp[i][j]$表示i这棵子树内所有的点都在某条路径上，且通过i点的路径最高向上延申到深度为j的地方。$dp[i][j] = min_{min(j_1, j_2, ..., j_n)=j} (dp[c_1][j_1] + dp[c_2][j_2] + ... + dp[c_n][j_n])$。

```cpp
// O(nm)
#include <bits/stdc++.h>
using namespace std;

typedef long long ll;

const int N = 2010;
const ll inf = 1ll << 60;

int n, m;
vector<int> son[N];
vector<array<int, 2>> path[N];
int dep[N];

ll dp[N][N];

void merge(ll *a, ll *b, int len) {
    static ll sufa[N], sufb[N];
    sufa[len + 1] = inf;
    sufb[len + 1] = inf;
    for (int i = len; i >= 1; i--) {
        sufa[i] = min(sufa[i + 1], a[i]);
        sufb[i] = min(sufb[i + 1], b[i]);
    }
    for (int i = 1; i <= len; i++) {
        a[i] = min(a[i] + sufb[i], b[i] + sufa[i]);
    }
}

void dfs(int u) {
    for (int i = 1; i <= dep[u]; i++) dp[u][i] = inf;
    for (auto p : path[u]) {
        dp[u][dep[p[0]]] = min(dp[u][dep[p[0]]], (ll)p[1]);
    }
    for (auto v : son[u]) {
        dfs(v);
        merge(dp[u], dp[v], dep[v]);
    }
}

int main() {
    cin >> n >> m;
    dep[1] = 1;
    for (int i = 2; i <= n; i++) {
        int f;
        cin >> f;
        son[f].push_back(i);
        dep[i] = dep[f] + 1; // 因为输入是按照顺序的，这里这样也行
    }
    for (int i = 1; i <= m; i++) {
        int u, v, a;
        cin >> u >> v >> a;
        path[v].push_back({u, a});
    }
    dfs(1);
    if (dp[1][1] >= inf)
        cout << -1;
    else
        cout << dp[1][1];
    return 0;
}
```

## 树上连通块
给定一个n个点的树。对于每个点，求出包含这个点的连通块的个数，答案对m取模。

如果令$dp[i]$表示i这棵子树中包含i节点的连通块的个数，很简单得出$dp[i]=\Pi _{c_i\in childs[i]}(dp[c_i]+1)$。直接换根dp的话由于答案对m取模，父节点dp除以子节点dp不太对，所以我们引入一个dp2数组表示去除某个节点的子节点后包含该节点的连通块的个数。则最终答案$ans[i]=(dp2[father] + 1) * dp[i] % mod$。而$dp2[i]=(dp2[father] + 1) * \Pi (dp[brothers] + 1) % mod$。

```cpp
#include <bits/stdc++.h>
using namespace std;

typedef long long ll;

const int N = 2010;
const ll inf = 1ll << 60;

int n, mod;
vector<int> son[N];

ll dp[N], dp2[N], ans[N];

void dfs1(int u) {
    dp[u] = 1;
    for (auto v : son[u]) {
        dfs1(v);
        dp[u] = dp[u] * (dp[v] + 1) % mod;
    }
}

void dfs2(int u) {
    static ll pre[N], suf[N];
    int m = son[u].size();
    if (m == 0) return;
    pre[0] = 1;
    for (int i = 0; i < m; i++) {
        int v = son[u][i];
        pre[i + 1] = pre[i] * (dp[v] + 1) % mod;
    }
    suf[m] = 1;
    for (int i = m - 1; i >= 0; i--) {
        int v = son[u][i];
        suf[i] = suf[i + 1] * (dp[v] + 1) % mod;
    }
    for (int i = 0; i < m; i++) {
        int v = son[u][i];
        dp2[v] = pre[i] * suf[i + 1] % mod;
        if (u != 1) dp2[v] = dp2[v] * (dp2[u] + 1) % mod;
    }
    for (auto v : son[u]) {
        ans[v] = (dp2[v] + 1) * dp[v] % mod;
        dfs2(v);
    }
}

int main() {
    cin >> n >> mod;
    for (int i = 2; i <= n; i++) {
        int f;
        cin >> f;
        son[f].push_back(i);
    }
    dfs1(1);
    dfs2(1);
    ans[1] = dp[1];
    for (int i = 1; i <= n; i++)
        cout << ans[i] << endl;
    return 0;
}
```
