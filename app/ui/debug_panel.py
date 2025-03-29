"""
Debug panel for visualizing agent behavior in the Streamlit UI.
"""
import logging
from typing import Dict, List, Any, Optional

import streamlit as st

logger = logging.getLogger(__name__)

def create_debug_panel(state: Dict[str, Any]):
    """Create debug panel to visualize agent behavior.
    
    Args:
        state (Dict[str, Any]): Current workflow state
    """
    st.subheader("Agent Debugging")
    
    if not state:
        st.info("No agent activity to display")
        return
    
    # Display current processing state
    if state.get("processing", False):
        st.info("Processing in progress...")
    
    # Show tutorial plan if available
    if "plan" in state and state["plan"]:
        with st.expander("Tutorial Plan", expanded=True):
            _display_tutorial_plan(state)
    
    # Show agent observations if available
    if "observations" in state and state["observations"]:
        with st.expander("Agent Observations", expanded=True):
            _display_agent_observations(state)

def _display_tutorial_plan(state: Dict[str, Any]):
    """Display the tutorial plan.
    
    Args:
        state (Dict[str, Any]): Current workflow state
    """
    plan = state["plan"]
    current_section = state.get("current_section", "")
    
    for i, section in enumerate(plan):
        if section == current_section:
            st.markdown(f"**→ {section}**")
        else:
            st.markdown(f"- {section}")

def _display_agent_observations(state: Dict[str, Any]):
    """Display agent observations.
    
    Args:
        state (Dict[str, Any]): Current workflow state
    """
    observations = state["observations"]
    
    # Display only the most recent observation for simplicity
    if observations:
        last_obs = observations[-1]
        
        # Agent type
        st.subheader(f"Agent: {last_obs.get('agent', 'Unknown')}")
        
        # Thoughts
        thoughts = last_obs.get("thoughts", [])
        if thoughts:
            with st.expander("Thoughts", expanded=True):
                st.write(thoughts[-1])  # Show the most recent thought
        
        # Actions
        actions = last_obs.get("actions", [])
        if actions:
            with st.expander("Actions", expanded=True):
                action = actions[-1]  # Show the most recent action
                tool = action.get("tool", "Unknown")
                st.write(f"Tool: {tool}")
                
                # Show input if not too large
                input_data = action.get("input", {})
                if isinstance(input_data, dict) and input_data:
                    # Truncate any large string values
                    for k, v in input_data.items():
                        if isinstance(v, str) and len(v) > 200:
                            input_data[k] = v[:200] + "..."
                    st.json(input_data)
                elif isinstance(input_data, str):
                    if len(input_data) > 200:
                        st.write(f"Input: {input_data[:200]}...")
                    else:
                        st.write(f"Input: {input_data}")
        
        # Observations from actions
        action_observations = last_obs.get("observations", [])
        if action_observations:
            with st.expander("Observations", expanded=True):
                obs = action_observations[-1]  # Show the most recent observation
                
                # Handle different types of observations
                if isinstance(obs, dict):
                    # Truncate any large string values
                    for k, v in obs.items():
                        if isinstance(v, str) and len(v) > 200:
                            obs[k] = v[:200] + "..."
                    st.json(obs)
                elif isinstance(obs, str):
                    if len(obs) > 200:
                        st.write(obs[:200] + "...")
                    else:
                        st.write(obs)
                elif isinstance(obs, list):
                    st.write(f"List with {len(obs)} items")
                else:
                    st.write("Observation data not displayable")
        
        # Result
        if "result" in last_obs:
            with st.expander("Result", expanded=True):
                result = last_obs["result"]
                if isinstance(result, str):
                    if len(result) > 200:
                        st.write(result[:200] + "...")
                    else:
                        st.write(result)
                elif isinstance(result, list):
                    st.write("\n".join([f"- {item}" for item in result[:5]]))
                    if len(result) > 5:
                        st.write(f"... and {len(result) - 5} more items")
                elif isinstance(result, dict):
                    st.json(result)

def display_react_debugging(agent_type: str, thought: str, action: Dict[str, Any], observation: Any):
    """Display ReAct debugging information.
    
    Args:
        agent_type (str): Type of agent
        thought (str): Agent thought
        action (Dict[str, Any]): Agent action
        observation (Any): Observation from action
    """
    st.subheader(f"{agent_type} Agent")
    
    # Display thought
    with st.expander("Thought", expanded=True):
        st.write(thought)
    
    # Display action
    with st.expander("Action", expanded=True):
        st.write(f"Tool: {action.get('tool', 'Unknown')}")
        st.json(action.get("input", {}))
    
    # Display observation
    with st.expander("Observation", expanded=True):
        if isinstance(observation, dict):
            st.json(observation)
        elif isinstance(observation, str):
            st.write(observation)
        else:
            st.write(str(observation))
