#include <bits/stdc++.h>
using namespace std;

template <class T>
struct Edge
{
    int to;
    T weight;
    Edge(int t, T w)
    {
        to = t;
        weight = w;
    }
};

template <class T, int SCALE>
struct Graph
{
    std::vector<T> g[SCALE];
    void addEdge(int from, T to)
    {
        g[from].push_back(to);
    }
    std::vector<T> &operator[](int node)
    {
        return g[node];
    }
};

const int SCALE = 1e5 + 5;
Graph<int, SCALE> g;

struct Tarjan
{
    int n;
    vector<int> dfn;
    vector<int> low;
    int dfncnt;
    stack<int> s;
    vector<int> in_stack;
    vector<int> scc_no;
    int sccnt;
    vector<int> scc_size;

    Tarjan(int n)
    {
        this->n = n;
        dfn.resize(n + 1);
        dfn.assign(dfn.size(), 0);
        low.resize(n + 1);
        low.assign(low.size(), 0);
        in_stack.resize(n + 1);
        in_stack.assign(in_stack.size(), 0);
        scc_no.resize(n + 1);
        scc_no.assign(scc_no.size(), 0);
        scc_size.resize(n + 1);
        scc_size.assign(scc_size.size(), 0);
        dfncnt = 0;
        sccnt = 0;
    }

    void dfs(int u)
    {
        dfn[u] = low[u] = ++dfncnt;
        s.push(u);
        in_stack[u] = 1;
        for (auto to : g[u])
        {
            if (!dfn[to])
            {
                dfs(to);
                low[u] = min(low[u], low[to]);
            }
            else if (in_stack[to])
            {
                low[u] = min(low[u], dfn[to]);
            }
        }
        if (dfn[u] == low[u])
        {
            sccnt++;
            while (s.top() != u)
            {
                scc_no[s.top()] = sccnt;
                in_stack[s.top()] = 0;
                s.pop();
                scc_size[sccnt]++;
            }
            scc_no[u] = sccnt;
            in_stack[u] = 0;
            scc_size[sccnt]++;
            s.pop();
        }
    }

    vector<int> &tarjan() {
        for (int i = 1; i <= n; i++) {
            if (!scc_no[i])
                dfs(i);
        }
        return scc_no;
    }

};

int n, m;

int main()
{
    cin >> n >> m;
    for (int i = 0; i < m; i++)
    {
        int from, to;
        cin >> from >> to;
        g.addEdge(from, to);
    }
    Tarjan tarjan(n);
    auto &scc_no = tarjan.tarjan();
    map<int, vector<int>> sccno2nodes;
    for (int i = 1; i <= n; i++)
    {
        sccno2nodes[scc_no[i]].push_back(i);
    }
    cout << sccno2nodes.size() << endl;
    for (auto pa : sccno2nodes)
    {
        cout << pa.second.size() << " ";
        for (auto node : pa.second)
            cout << node << " ";
        cout << endl;
    }
    return 0;
}