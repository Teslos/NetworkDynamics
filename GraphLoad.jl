#using ParserCombinator
using Graphs
using GraphIO
using GraphIO.GML # for GMLFormat
#using GraphIO.GraphML # for GraphMLFormat
#using GraphIO.NET # for NETFormat
load_path = "C:\\Users\\ivt\\Documents\\CODE\\Duffing\\VS_VERSION_GUI\\Osc_GUI\\Osc_GUI\\graph.gml"
#G = loadgraph(load_path, "graph6", )
#G = loadgraph(load_path, NETFormat())
load_path = "C:\\Users\\ivt\\Documents\\CODE\\learning\\network-py\\grid_2d_graph.gml"
if isfile(load_path)
    G = loadgraph(load_path, "graph", GMLFormat())
else
    println("File does not exist")
end
print(G)
# plot the graph
using GraphMakie, GLMakie
using GraphMakie.NetworkLayout

graphplot(G)