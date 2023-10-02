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
vector<int> dis(SCALE, 0);
Graph<Edge<int>, SCALE> g;

#define INF 0x3FFFFFFF
void spfa(int s, int n)
{
    vector<int> vis(n + 1, 0);
    for (int i = 1; i <= n; i++)
        dis[i] = (i == s ? 0 : INF);
    queue<int> q;
    q.push(s);
    vis[s] = 1;
    while (!q.empty())
    {
        int v = q.front();
        q.pop();
        vis[v] = 0;
        for (int i = 0; i < g[v].size(); i++)
        {
            int t = g[v][i].to;
            int w = g[v][i].weight;
            if (dis[t] > dis[v] + w)
            {
                dis[t] = dis[v] + w;
                if (vis[t] == 0)
                    q.push(t);
                vis[t] = 1;
            }
        }
    }
}

int n, m, s;

int main()
{
    cin >> n >> m >> s;
    for (int i = 0; i < m; i++)
    {
        int from, to, weight;
        cin >> from >> to >> weight;
        g.addEdge(from, Edge<int>(to, weight));
    }
    spfa(s, n);
    for (int i = 1; i <= n; i++)
        cout << (dis[i] == INF ? 2147483647 : dis[i]) << " ";
    return 0;
}