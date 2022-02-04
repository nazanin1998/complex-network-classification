from cnn_model.run_cnn_phase import run_cnn_phase
from graph_generate.generate_graph import GraphGenerator

# todo call main class of first phase => Negin
graph_generator = GraphGenerator(graph_count=10)
graph_generator.generate_graphs()

# todo call main class of second phase => Sarvin

# todo call main class of third phase => Nazanin
run_cnn_phase()
