from sim import Economy
from utils.simulationObjects import Config
from mesa.mesa_logging import DEBUG, log_to_stderr
from mesa.visualization import (
    SolaraViz,
    make_plot_component,
)

log_to_stderr(DEBUG)

# model_params = {
#     "seed": {
#         "type": "InputText",
#         "value": 42,
#         "label": "Random Seed",
#     },
#     "n": {
#         "type": "SliderInt",
#         "value": 50,
#         "label": "Number of agents:",
#         "min": 10,
#         "max": 100,
#         "step": 1,
#     },
#     "width": 10,
#     "height": 10,
# }


def post_process(ax):
    ax.get_figure().colorbar(ax.collections[0], label="wealth", ax=ax)


# Create initial model instance
model = Economy(Config(
    NUM_INDIVIDUAL=100,
    NUM_COMPANY=5,
    SEED=42,
    # FUNDS_PRECISION = int,
))

# Create visualization elements. The visualization elements are solara components
# that receive the model instance as a "prop" and display it in a certain way.
# Under the hood these are just classes that receive the model instance.
# You can also author your own visualization elements, which can also be functions
# that receive the model instance and return a valid solara component.

# SpaceGraph = make_space_component(
#     agent_portrayal, cmap="viridis", vmin=0, vmax=10, post_process=post_process
# )
GiniPlot = make_plot_component("individual_wealth_gini")


# Create the SolaraViz page. This will automatically create a server and display the
# visualization elements in a web browser.
# Display it using the following command in the example directory:
# solara run app.py
# It will automatically update and display any changes made to this file
page = SolaraViz(
    model,
    components=[GiniPlot],
    # model_params=model_params,
    name="ECO SIM",
)
page  # noqa


# In a notebook environment, we can also display the visualization elements directly
# SpaceGraph(model1)
# GiniPlot(model1)

# The plots will be static. If you want to pick up model steps,
# you have to make the model reactive first
# reactive_model = solara.reactive(model1)
# SpaceGraph(reactive_model)
# In a different notebook block:
# reactive_model.value.step()