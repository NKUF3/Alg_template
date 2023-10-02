#ifndef __GRAPH_H_
#define __GRAPH_H_
#include <bits/stdc++.h>

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

#endif