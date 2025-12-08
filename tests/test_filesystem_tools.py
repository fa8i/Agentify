import os
import tempfile
import pytest
from agentify.tools.filesystem import ListDirTool, ReadFileTool, WriteFileTool


class TestFilesystemTools:
    """Test suite for filesystem tools."""

    def test_list_dir_tool(self):
        """Test ListDirTool enumerates directory contents."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files
            open(os.path.join(tmpdir, "file1.txt"), "w").close()
            open(os.path.join(tmpdir, "file2.txt"), "w").close()
            os.makedirs(os.path.join(tmpdir, "subdir"))

            tool = ListDirTool(sandbox_dir=tmpdir)
            result = tool._list_dir(".")
            
            assert "file1.txt" in result
            assert "file2.txt" in result
            assert "subdir/" in result

    def test_read_file_tool(self):
        """Test ReadFileTool reads file contents."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = os.path.join(tmpdir, "test.txt")
            test_content = "Hello, Deep Agent!"
            
            with open(test_file, "w") as f:
                f.write(test_content)
            
            tool = ReadFileTool(sandbox_dir=tmpdir)
            result = tool._read_file("test.txt")
            
            assert result == test_content

    def test_write_file_tool(self):
        """Test WriteFileTool creates and writes files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tool = WriteFileTool(sandbox_dir=tmpdir)
            result = tool._write_file("output.txt", "Test content")
            
            assert "Successfully wrote" in result
            
            # Verify file exists and has correct content
            with open(os.path.join(tmpdir, "output.txt"), "r") as f:
                assert f.read() == "Test content"

    def test_sandbox_security(self):
        """Test that sandbox prevents access outside its boundaries."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tool = ReadFileTool(sandbox_dir=tmpdir)
            
            # Try to read outside sandbox
            result = tool._read_file("../../etc/passwd")
            assert "Access denied" in result or "Error" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
