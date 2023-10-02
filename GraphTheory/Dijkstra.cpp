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
typedef pair<int, int> PII;
typedef vector<PII> VII;
void dijkstra(int s, int n)
{
    vector<int> vis(n + 1, 0);
    for (int i = 1; i <= n; i++)
        dis[i] = (i == s ? 0 : INF);
    // greater实现小顶堆, less 实现大顶堆（默认为大顶堆）
    priority_queue<PII, VII, greater<PII>> q;
    q.push(make_pair(dis[s], s));
    while (!q.empty())
    {
        PII p = q.top();
        q.pop();
        int x = p.second;
        if (vis[x])
            continue;
        vis[x] = 1;
        for (int i = 0; i < g[x].size(); i++)
        {
            int y = g[x][i].to;
            int d = g[x][i].weight;
            if (!vis[y] && dis[x] + d < dis[y])
            {
                dis[y] = dis[x] + d;
                q.push(make_pair(dis[y], y));
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
    dijkstra(s, n);
    for (int i = 1; i <= n; i++)
        cout << dis[i] << " ";
    return 0;
}