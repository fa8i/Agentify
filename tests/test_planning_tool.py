import pytest
from agentify.tools.planning import TodoTool


class TestPlanningTool:
    """Test suite for planning tool."""

    def test_add_task(self):
        """Test adding tasks to the plan."""
        tool = TodoTool()
        result = tool._manage_todos(action="add", task="Complete the demo")
        
        assert "Task added" in result
        assert "[0]" in result
        assert "Complete the demo" in result

    def test_list_tasks(self):
        """Test listing all tasks."""
        tool = TodoTool()
        tool._manage_todos(action="add", task="Task 1")
        tool._manage_todos(action="add", task="Task 2")
        
        result = tool._manage_todos(action="list")
        
        assert "Task 1" in result
        assert "Task 2" in result
        assert "[ ]" in result  # Pending status

    def test_complete_task(self):
        """Test marking a task as complete."""
        tool = TodoTool()
        tool._manage_todos(action="add", task="Do something")
        
        result = tool._manage_todos(action="complete", task_id=0)
        assert "marked as completed" in result
        
        list_result = tool._manage_todos(action="list")
        assert "[x]" in list_result  # Completed status

    def test_remove_task(self):
        """Test removing a task."""
        tool = TodoTool()
        tool._manage_todos(action="add", task="Task to remove")
        tool._manage_todos(action="add", task="Task to keep")
        
        result = tool._manage_todos(action="remove", task_id=0)
        assert "removed" in result
        
        list_result = tool._manage_todos(action="list")
        assert "Task to remove" not in list_result
        assert "Task to keep" in list_result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
