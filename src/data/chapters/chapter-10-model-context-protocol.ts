import { Chapter } from '../types'

export const modelContextProtocolChapter: Chapter = {
  id: 'model-context-protocol',
  number: 10,
  title: 'Model Context Protocol',
  part: 'Part Two – Learning and Adaptation',
  description: 'Standardize LLM-external system communication through MCP, enabling universal interfaces for tools, resources, and data integration across diverse operational environments.',
  readingTime: '28 min read',
  difficulty: 'Advanced',
  content: {
    overview: `The Model Context Protocol (MCP) represents a paradigm shift in how Large Language Models interface with external systems, providing a standardized, universal adapter that enables any LLM to seamlessly integrate with any external application, database, or tool without requiring custom integrations for each connection.

Operating on a client-server architecture, MCP standardizes the exposure and consumption of three critical components: resources (static data like files and database records), tools (executable functions that perform actions), and prompts (interactive templates that guide LLM interactions). This standardization dramatically reduces integration complexity while promoting interoperability, composability, and reusability across different systems and implementations.

MCP transforms the traditional one-to-one, proprietary approach of tool function calling into an open ecosystem where compliant tools can be dynamically discovered and accessed by any compliant LLM. By adopting a federated model, MCP enables organizations to bring disparate legacy services into modern AI workflows simply by wrapping them in MCP-compliant interfaces, preserving existing investments while unlocking new capabilities.

However, MCP's effectiveness depends heavily on the design of underlying APIs and data formats. Simply wrapping legacy APIs without optimization can result in suboptimal agent performance. The protocol's true power emerges when combined with agent-friendly APIs that provide deterministic features like filtering and sorting, and when data is presented in formats that agents can effectively process and understand.`,

    keyPoints: [
      'Provides universal standardization for LLM-external system communication, eliminating the need for custom integrations between each LLM and external tool',
      'Implements client-server architecture with dynamic discovery capabilities, allowing agents to identify and access available tools, resources, and prompts without redeployment',
      'Supports multiple transport mechanisms including JSON-RPC over STDIO for local interactions and HTTP/SSE for remote server communication',
      'Enables federated integration of legacy systems through MCP-compliant wrappers, preserving existing infrastructure while enabling modern AI workflows',
      'Distinguishes between resources (static data), tools (executable functions), and prompts (interaction templates) for comprehensive external system integration',
      'Requires agent-friendly API design and data formats to achieve optimal performance, as MCP itself does not guarantee data compatibility or API optimization',
      'Integrates seamlessly with modern agent frameworks like Google ADK, providing built-in MCPToolset support for rapid development and deployment',
      'Facilitates complex workflow orchestration by combining multiple MCP-exposed services for multi-step, cross-system automation and data processing'
    ],

    codeExample: `# Complete MCP Implementation: Agent Development Kit Integration
import os
import asyncio
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime
import json

# Google ADK imports for MCP integration
from google.adk.agents import LlmAgent
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, StdioServerParameters, HttpServerParameters

# FastMCP imports for server creation
from fastmcp import FastMCP, Client
from fastmcp.server import ServerConfig
from fastmcp.types import Resource, Tool, Prompt

@dataclass
class MCPConnectionConfig:
    """Configuration for MCP server connections."""
    name: str
    connection_type: str  # 'stdio' or 'http'
    server_params: Dict[str, Any]
    tool_filter: Optional[List[str]] = None
    description: str = ""

class ComprehensiveMCPAgent:
    """
    Advanced MCP integration demonstrating multiple connection types,
    server management, and complex workflow orchestration.
    """
    
    def __init__(self, agent_name: str = "comprehensive_mcp_agent"):
        """
        Initialize comprehensive MCP agent with multiple server connections.
        
        Args:
            agent_name: Unique identifier for the agent instance
        """
        self.agent_name = agent_name
        self.target_folder = self._setup_managed_directory()
        self.mcp_servers = {}
        self.connection_configs = []
        self.agent = None
        
        print(f"🤖 Initializing Comprehensive MCP Agent: {agent_name}")
        print(f"📁 Managed Directory: {self.target_folder}")
    
    def _setup_managed_directory(self) -> str:
        """Create and return path to managed file directory."""
        target_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 
            "mcp_managed_files"
        )
        os.makedirs(target_path, exist_ok=True)
        
        # Create sample files for demonstration
        sample_files = {
            "sample.txt": "This is a sample text file managed by MCP.",
            "data.json": json.dumps({"users": [{"name": "Alice", "role": "admin"}, {"name": "Bob", "role": "user"}], "timestamp": datetime.now().isoformat()}),
            "readme.md": "# MCP Managed Files\\n\\nThis directory is managed by the MCP filesystem server.\\n\\n## Features\\n- File listing\\n- Content reading\\n- File writing"
        }
        
        for filename, content in sample_files.items():
            file_path = os.path.join(target_path, filename)
            if not os.path.exists(file_path):
                with open(file_path, 'w') as f:
                    f.write(content)
        
        return target_path
    
    def add_filesystem_connection(self) -> 'ComprehensiveMCPAgent':
        """Add filesystem MCP server connection."""
        
        config = MCPConnectionConfig(
            name="filesystem_server",
            connection_type="stdio",
            server_params={
                "command": "npx",
                "args": [
                    "-y",
                    "@modelcontextprotocol/server-filesystem",
                    self.target_folder
                ]
            },
            tool_filter=["list_directory", "read_file", "write_file"],
            description="Filesystem operations for file management and content manipulation"
        )
        
        self.connection_configs.append(config)
        print(f"📂 Added filesystem server connection: {config.name}")
        return self
    
    def add_database_connection(self, db_path: str = None) -> 'ComprehensiveMCPAgent':
        """Add database MCP server connection."""
        
        if db_path is None:
            db_path = os.path.join(self.target_folder, "sample.db")
            self._create_sample_database(db_path)
        
        config = MCPConnectionConfig(
            name="database_server", 
            connection_type="stdio",
            server_params={
                "command": "npx",
                "args": [
                    "-y",
                    "@modelcontextprotocol/server-sqlite",
                    db_path
                ]
            },
            tool_filter=["query", "execute", "list_tables"],
            description="SQLite database operations for data querying and manipulation"
        )
        
        self.connection_configs.append(config)
        print(f"🗄️ Added database server connection: {config.name}")
        return self
    
    def add_custom_fastmcp_connection(self, server_url: str = "http://localhost:8000") -> 'ComprehensiveMCPAgent':
        """Add custom FastMCP HTTP server connection."""
        
        config = MCPConnectionConfig(
            name="custom_fastmcp_server",
            connection_type="http", 
            server_params={
                "url": server_url
            },
            tool_filter=["greet", "calculate", "analyze_text"],
            description="Custom FastMCP server with specialized business logic tools"
        )
        
        self.connection_configs.append(config)
        print(f"🌐 Added FastMCP server connection: {config.name}")
        return self
    
    def _create_sample_database(self, db_path: str):
        """Create sample SQLite database for demonstration."""
        import sqlite3
        
        if os.path.exists(db_path):
            return
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Create sample tables
        cursor.execute('''
            CREATE TABLE users (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                role TEXT DEFAULT 'user',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE projects (
                id INTEGER PRIMARY KEY,
                title TEXT NOT NULL,
                description TEXT,
                status TEXT DEFAULT 'active',
                user_id INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        # Insert sample data
        sample_users = [
            ('Alice Johnson', 'alice@example.com', 'admin'),
            ('Bob Smith', 'bob@example.com', 'user'),
            ('Carol Davis', 'carol@example.com', 'manager')
        ]
        
        cursor.executemany('INSERT INTO users (name, email, role) VALUES (?, ?, ?)', sample_users)
        
        sample_projects = [
            ('MCP Integration', 'Implement Model Context Protocol', 'active', 1),
            ('Agent Framework', 'Build advanced agent capabilities', 'in_progress', 2),
            ('Documentation', 'Create comprehensive documentation', 'completed', 3)
        ]
        
        cursor.executemany('INSERT INTO projects (title, description, status, user_id) VALUES (?, ?, ?, ?)', sample_projects)
        
        conn.commit()
        conn.close()
        print(f"📊 Created sample database: {db_path}")
    
    def build_agent(self) -> 'ComprehensiveMCPAgent':
        """Build the ADK agent with all configured MCP connections."""
        
        if not self.connection_configs:
            raise ValueError("No MCP connections configured. Add at least one connection before building agent.")
        
        # Create MCPToolset instances for each configuration
        mcp_toolsets = []
        
        for config in self.connection_configs:
            if config.connection_type == "stdio":
                connection_params = StdioServerParameters(
                    command=config.server_params["command"],
                    args=config.server_params["args"],
                    env=config.server_params.get("env", {})
                )
            elif config.connection_type == "http":
                connection_params = HttpServerParameters(
                    url=config.server_params["url"]
                )
            else:
                raise ValueError(f"Unsupported connection type: {config.connection_type}")
            
            toolset = MCPToolset(
                connection_params=connection_params,
                tool_filter=config.tool_filter
            )
            
            mcp_toolsets.append(toolset)
            print(f"🔧 Configured toolset: {config.name}")
        
        # Create comprehensive agent instruction
        instruction = f'''You are {self.agent_name}, an advanced AI agent with comprehensive MCP capabilities.

**Your Primary Functions:**
- File System Management: List, read, and write files in the managed directory
- Database Operations: Query and manipulate SQLite database records  
- Custom Tool Integration: Use specialized tools via FastMCP servers
- Complex Workflow Orchestration: Combine multiple MCP services for multi-step tasks

**Available MCP Connections:**
{self._generate_connection_summary()}

**Operational Guidelines:**
1. Always verify available tools before attempting operations
2. Provide detailed explanations of actions taken
3. Handle errors gracefully and suggest alternatives
4. Combine multiple MCP services when beneficial for complex tasks
5. Maintain data consistency across operations

**File System Context:**
- Managed Directory: {self.target_folder}
- Sample files available for demonstration and testing
- Can create, read, and modify files as needed

**Database Context:**
- SQLite database with users and projects tables
- Sample data available for querying and analysis
- Support for complex SQL operations and reporting

Use your MCP capabilities to provide comprehensive assistance with data management, file operations, and custom tool integration.'''
        
        # Build the agent
        self.agent = LlmAgent(
            model='gemini-2.0-flash',
            name=self.agent_name,
            instruction=instruction,
            tools=mcp_toolsets
        )
        
        print(f"🚀 Built comprehensive MCP agent with {len(mcp_toolsets)} toolsets")
        return self
    
    def _generate_connection_summary(self) -> str:
        """Generate summary of configured connections for agent instruction."""
        summary_lines = []
        for i, config in enumerate(self.connection_configs, 1):
            summary_lines.append(f"{i}. {config.name} ({config.connection_type}): {config.description}")
        return "\\n".join(summary_lines)
    
    async def demonstrate_capabilities(self):
        """Demonstrate comprehensive MCP capabilities through example interactions."""
        
        if not self.agent:
            raise ValueError("Agent not built. Call build_agent() first.")
        
        print("\\n🎯 DEMONSTRATING COMPREHENSIVE MCP CAPABILITIES")
        print("="*60)
        
        # Demonstration scenarios
        scenarios = [
            {
                "name": "File System Exploration",
                "prompt": "List all files in the managed directory and read the contents of sample.txt"
            },
            {
                "name": "Database Analysis", 
                "prompt": "Query the database to show all users and their associated projects, including project status"
            },
            {
                "name": "Cross-System Integration",
                "prompt": "Create a summary report by combining database user information with file system content, then save it as a new file"
            },
            {
                "name": "Custom Tool Usage",
                "prompt": "Use any custom tools to greet the database users and perform text analysis on the readme.md file"
            }
        ]
        
        for scenario in scenarios:
            print(f"\\n📝 Scenario: {scenario['name']}")
            print(f"Prompt: {scenario['prompt']}")
            print("-" * 40)
            
            try:
                # In a real implementation, you would interact with the agent here
                # For demonstration, we'll show the expected flow
                print("🤖 Agent would:")
                print("   1. Analyze the request and identify required MCP tools")
                print("   2. Execute appropriate tool calls across multiple MCP servers")
                print("   3. Synthesize results from different data sources")  
                print("   4. Provide comprehensive response with detailed explanations")
                print("   ✅ Scenario completed successfully")
                
                await asyncio.sleep(0.5)  # Simulate processing time
                
            except Exception as e:
                print(f"   ❌ Error in scenario: {str(e)}")
    
    def get_configuration_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of MCP agent configuration."""
        
        return {
            "agent_name": self.agent_name,
            "managed_directory": self.target_folder,
            "connection_count": len(self.connection_configs),
            "connections": [
                {
                    "name": config.name,
                    "type": config.connection_type, 
                    "tools": config.tool_filter or "all",
                    "description": config.description
                }
                for config in self.connection_configs
            ],
            "capabilities": [
                "File system operations (list, read, write)",
                "Database querying and manipulation",
                "Custom tool integration via HTTP/FastMCP",
                "Multi-server workflow orchestration",
                "Dynamic tool discovery and usage"
            ]
        }

# FastMCP Server Implementation
class AdvancedFastMCPServer:
    """
    Comprehensive FastMCP server with multiple tools, resources, and prompts.
    Demonstrates advanced MCP server capabilities and patterns.
    """
    
    def __init__(self, host: str = "127.0.0.1", port: int = 8000):
        """
        Initialize advanced FastMCP server.
        
        Args:
            host: Server host address
            port: Server port number
        """
        self.host = host
        self.port = port
        self.server = FastMCP()
        self.data_store = {}
        
        self._register_tools()
        self._register_resources() 
        self._register_prompts()
        
        print(f"🌐 Advanced FastMCP Server initialized on {host}:{port}")
    
    def _register_tools(self):
        """Register comprehensive set of MCP tools."""
        
        @self.server.tool
        def greet(name: str, title: str = "friend") -> str:
            """
            Generate a personalized greeting with optional title.
            
            Args:
                name: The person's name to greet
                title: Optional title or role (default: "friend")
                
            Returns:
                Personalized greeting string
            """
            return f"Hello, {name}! Nice to meet you, {title}."
        
        @self.server.tool
        def calculate(expression: str) -> Dict[str, Any]:
            """
            Safely evaluate mathematical expressions and return results.
            
            Args:
                expression: Mathematical expression to evaluate (e.g., "2 + 3 * 4")
                
            Returns:
                Dictionary with result and metadata
            """
            try:
                # Safe evaluation of basic mathematical expressions
                allowed_chars = set('0123456789+-*/.() ')
                if not all(c in allowed_chars for c in expression):
                    return {
                        "error": "Expression contains invalid characters",
                        "allowed": "Numbers, +, -, *, /, ., (, ), and spaces only"
                    }
                
                result = eval(expression)
                return {
                    "expression": expression,
                    "result": result,
                    "type": type(result).__name__,
                    "status": "success"
                }
            except Exception as e:
                return {
                    "expression": expression,
                    "error": str(e),
                    "status": "error"
                }
        
        @self.server.tool
        def analyze_text(text: str, analysis_type: str = "comprehensive") -> Dict[str, Any]:
            """
            Perform text analysis with various metrics and insights.
            
            Args:
                text: Text content to analyze
                analysis_type: Type of analysis ("basic", "comprehensive", "sentiment")
                
            Returns:
                Dictionary with analysis results and metrics
            """
            basic_stats = {
                "character_count": len(text),
                "word_count": len(text.split()),
                "line_count": len(text.splitlines()),
                "paragraph_count": len([p for p in text.split('\\n\\n') if p.strip()])
            }
            
            if analysis_type == "basic":
                return {"analysis_type": "basic", "stats": basic_stats}
            
            # Comprehensive analysis
            words = text.lower().split()
            word_freq = {}
            for word in words:
                clean_word = word.strip('.,!?";()[]')
                if clean_word:
                    word_freq[clean_word] = word_freq.get(clean_word, 0) + 1
            
            top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
            
            comprehensive_stats = {
                **basic_stats,
                "unique_words": len(word_freq),
                "average_word_length": sum(len(word) for word in words) / len(words) if words else 0,
                "top_words": top_words,
                "readability_estimate": "medium"  # Simplified estimate
            }
            
            if analysis_type == "sentiment":
                # Simple sentiment analysis (in production, use proper NLP library)
                positive_words = ["good", "great", "excellent", "amazing", "wonderful", "fantastic"]
                negative_words = ["bad", "terrible", "awful", "horrible", "poor", "disappointing"]
                
                positive_count = sum(1 for word in words if word in positive_words)
                negative_count = sum(1 for word in words if word in negative_words)
                
                if positive_count > negative_count:
                    sentiment = "positive"
                elif negative_count > positive_count:
                    sentiment = "negative"  
                else:
                    sentiment = "neutral"
                
                comprehensive_stats["sentiment"] = {
                    "overall": sentiment,
                    "positive_signals": positive_count,
                    "negative_signals": negative_count
                }
            
            return {
                "analysis_type": analysis_type,
                "stats": comprehensive_stats,
                "timestamp": datetime.now().isoformat()
            }
        
        @self.server.tool
        def store_data(key: str, value: str, category: str = "general") -> Dict[str, str]:
            """
            Store data in server's memory for later retrieval.
            
            Args:
                key: Unique identifier for the data
                value: Data content to store
                category: Optional category for organization
                
            Returns:
                Confirmation of storage operation
            """
            if category not in self.data_store:
                self.data_store[category] = {}
            
            self.data_store[category][key] = {
                "value": value,
                "stored_at": datetime.now().isoformat()
            }
            
            return {
                "key": key,
                "category": category,
                "status": "stored",
                "message": f"Data stored successfully in category '{category}'"
            }
        
        @self.server.tool
        def retrieve_data(key: str = None, category: str = None) -> Dict[str, Any]:
            """
            Retrieve stored data by key and/or category.
            
            Args:
                key: Specific key to retrieve (optional)
                category: Category to search within (optional)
                
            Returns:
                Retrieved data or summary of available data
            """
            if not self.data_store:
                return {"message": "No data stored", "available_categories": []}
            
            if category and key:
                # Retrieve specific key from specific category
                if category in self.data_store and key in self.data_store[category]:
                    return {
                        "key": key,
                        "category": category,
                        "data": self.data_store[category][key]
                    }
                else:
                    return {"error": f"Key '{key}' not found in category '{category}'"}
            
            elif category:
                # Retrieve all data from specific category
                if category in self.data_store:
                    return {
                        "category": category,
                        "data": self.data_store[category]
                    }
                else:
                    return {"error": f"Category '{category}' not found"}
            
            elif key:
                # Search for key across all categories
                found_data = []
                for cat, data in self.data_store.items():
                    if key in data:
                        found_data.append({
                            "category": cat,
                            "key": key,
                            "data": data[key]
                        })
                
                if found_data:
                    return {"search_key": key, "results": found_data}
                else:
                    return {"error": f"Key '{key}' not found in any category"}
            
            else:
                # Return summary of all stored data
                summary = {}
                for category, data in self.data_store.items():
                    summary[category] = list(data.keys())
                
                return {
                    "summary": "All stored data",
                    "categories": summary,
                    "total_items": sum(len(data) for data in self.data_store.values())
                }
    
    def _register_resources(self):
        """Register MCP resources for static data access."""
        
        @self.server.resource(uri="system://server_info")
        def server_info() -> str:
            """Provide information about this MCP server."""
            info = {
                "server_name": "Advanced FastMCP Server",
                "version": "1.0.0",
                "host": self.host,
                "port": self.port,
                "capabilities": [
                    "Text processing and analysis",
                    "Mathematical calculations", 
                    "Data storage and retrieval",
                    "Personalized interactions"
                ],
                "startup_time": datetime.now().isoformat()
            }
            return json.dumps(info, indent=2)
        
        @self.server.resource(uri="data://stored_summary")
        def stored_data_summary() -> str:
            """Provide summary of currently stored data."""
            if not self.data_store:
                return json.dumps({"message": "No data currently stored"})
            
            summary = {
                "total_categories": len(self.data_store),
                "categories": {},
                "last_updated": datetime.now().isoformat()
            }
            
            for category, data in self.data_store.items():
                summary["categories"][category] = {
                    "item_count": len(data),
                    "keys": list(data.keys())
                }
            
            return json.dumps(summary, indent=2)
    
    def _register_prompts(self):
        """Register MCP prompts for guided interactions."""
        
        @self.server.prompt(name="data_analysis_assistant")
        def data_analysis_prompt(data_type: str = "text", focus: str = "comprehensive") -> str:
            """
            Generate a prompt template for data analysis tasks.
            
            Args:
                data_type: Type of data to analyze ("text", "numerical", "mixed")
                focus: Analysis focus ("comprehensive", "summary", "insights")
            """
            base_prompt = f"""You are a {data_type} data analysis assistant. Your task is to provide {focus} analysis.

**Analysis Guidelines:**
1. Examine the provided {data_type} data carefully
2. Identify key patterns, trends, and anomalies
3. Provide clear, actionable insights
4. Use appropriate statistical or analytical methods
5. Present results in an organized, understandable format

**Available Tools:**
- analyze_text: For detailed text analysis and metrics
- calculate: For mathematical operations and statistical calculations
- store_data/retrieve_data: For saving and accessing analysis results

**Output Format:**
- Executive Summary
- Detailed Findings
- Key Insights
- Recommendations (if applicable)
- Supporting Data and Calculations

Begin your analysis:"""

            return base_prompt
        
        @self.server.prompt(name="workflow_orchestrator")  
        def workflow_prompt(task_complexity: str = "medium") -> str:
            """
            Generate a prompt template for complex workflow orchestration.
            
            Args:
                task_complexity: Complexity level ("simple", "medium", "complex")
            """
            complexity_guidance = {
                "simple": "Focus on direct tool usage and straightforward task completion",
                "medium": "Combine multiple tools and consider data dependencies",
                "complex": "Design multi-step workflows with error handling and optimization"
            }
            
            guidance = complexity_guidance.get(task_complexity, complexity_guidance["medium"])
            
            workflow_prompt = f"""You are a workflow orchestration specialist handling {task_complexity} tasks.

**Orchestration Principles:**
{guidance}

**Available MCP Capabilities:**
- File system operations (read, write, list)
- Database queries and updates
- Text processing and analysis
- Mathematical calculations
- Data storage and retrieval

**Workflow Planning Steps:**
1. Analyze the request and identify required operations
2. Plan the sequence of MCP tool invocations
3. Consider data flow and dependencies between steps
4. Execute tools in optimal order
5. Synthesize results from multiple sources
6. Provide comprehensive response with detailed explanations

**Error Handling:**
- Anticipate potential failures at each step
- Provide alternative approaches when tools are unavailable
- Validate data and results at each stage
- Maintain data consistency throughout the workflow

Ready to orchestrate your workflow:"""

            return workflow_prompt
    
    async def run_server(self):
        """Start the FastMCP server with comprehensive capabilities."""
        print(f"🚀 Starting Advanced FastMCP Server on {self.host}:{self.port}")
        print(f"📊 Registered tools: {len(self.server._tools)} tools")
        print(f"📁 Registered resources: {len(self.server._resources)} resources")  
        print(f"💭 Registered prompts: {len(self.server._prompts)} prompts")
        
        await self.server.run(
            transport="http",
            host=self.host,
            port=self.port
        )

# Usage Examples and Demonstrations
async def demonstrate_comprehensive_mcp_integration():
    """
    Comprehensive demonstration of MCP capabilities with multiple servers,
    connection types, and workflow orchestration.
    """
    
    print("🌟 COMPREHENSIVE MCP INTEGRATION DEMONSTRATION")
    print("="*70)
    
    # Build comprehensive MCP agent
    agent_builder = ComprehensiveMCPAgent("comprehensive_demo_agent")
    
    agent = (agent_builder
             .add_filesystem_connection()
             .add_database_connection() 
             .add_custom_fastmcp_connection()
             .build_agent())
    
    # Display configuration
    config = agent.get_configuration_summary()
    print(f"\\n📋 AGENT CONFIGURATION SUMMARY:")
    print(f"Name: {config['agent_name']}")
    print(f"Connections: {config['connection_count']}")
    
    for conn in config['connections']:
        print(f"  • {conn['name']} ({conn['type']}): {conn['description']}")
    
    # Demonstrate capabilities
    await agent.demonstrate_capabilities()
    
    print("\\n✨ MCP Integration demonstration completed!")
    return agent

# FastMCP Server Startup
async def run_advanced_fastmcp_server():
    """Start the advanced FastMCP server for demonstrations."""
    
    server = AdvancedFastMCPServer()
    await server.run_server()

# Main execution example
if __name__ == "__main__":
    async def main():
        print("🔄 MCP COMPREHENSIVE DEMONSTRATION")
        print("="*50)
        
        # Option 1: Run comprehensive agent demonstration
        print("\\n1️⃣ AGENT DEMONSTRATION:")
        agent = await demonstrate_comprehensive_mcp_integration()
        
        # Option 2: Start FastMCP server (uncomment to run server)
        # print("\\n\\n2️⃣ FASTMCP SERVER:")
        # await run_advanced_fastmcp_server()
        
        print("\\n🎯 MCP demonstration completed successfully!")
    
    asyncio.run(main())`,

    practicalApplications: [
      '🗄️ Database Integration and Analytics: Enable agents to query Google BigQuery, PostgreSQL, and other databases for real-time data retrieval, report generation, and record updates through standardized MCP interfaces',
      '🎨 Generative Media Orchestration: Integrate Google Cloud generative services (Imagen for images, Veo for video, Chirp 3 HD for voice, Lyria for music) through MCP Tools for dynamic content creation workflows',  
      '🌐 External API Integration: Provide standardized access to weather APIs, financial data services, CRM systems, and social media platforms without requiring custom integration for each service',
      '🤖 IoT Device Control: Enable natural language control of smart home devices, industrial sensors, and robotics systems through MCP-compliant interfaces for seamless automation',
      '💰 Financial Services Automation: Connect agents to trading platforms, compliance systems, and market data feeds for automated analysis, trade execution, and regulatory reporting through secure MCP channels',
      '📊 Enterprise Workflow Orchestration: Combine multiple business systems (ERP, CRM, HR, inventory) through MCP servers for complex, multi-step automation and data synchronization across departments',
      '🔧 Development Tool Integration: Connect agents to CI/CD pipelines, code repositories, monitoring systems, and deployment platforms for automated DevOps operations and infrastructure management',
      '🎯 Reasoning-Based Information Extraction: Leverage LLM analytical capabilities through MCP to extract specific insights from large document sets, legal contracts, and technical specifications with contextual understanding'
    ],

    nextSteps: [
      'Set up Google ADK development environment with MCP toolset support for rapid agent development and testing',
      'Implement filesystem MCP server using @modelcontextprotocol/server-filesystem for document and file management capabilities',
      'Create custom FastMCP server with business-specific tools and integrate with existing enterprise systems and APIs',
      'Design secure MCP architecture with proper authentication, authorization, and access control for production deployments',
      'Explore database integration using MCP Toolbox for Databases to connect agents with organizational data sources',
      'Implement error handling and retry mechanisms for robust MCP client-server communication in production environments',
      'Develop MCP resource management for static data exposure including configuration files, templates, and reference materials',
      'Build comprehensive testing framework for MCP integrations including tool validation, server health monitoring, and performance benchmarking'
    ]
  },

  sections: [
    {
      title: 'MCP vs. Tool Function Calling: Architecture and Implementation Differences',
      content: `Understanding the fundamental differences between Model Context Protocol (MCP) and traditional tool function calling is crucial for making informed architectural decisions in agentic system design.

**Traditional Tool Function Calling: Direct Integration Approach**
Tool function calling represents a direct, one-to-one communication model between an LLM and specific, predefined functions:

**Characteristics of Function Calling:**
- **Proprietary Implementation**: Each LLM provider implements its own function calling format and protocol, creating vendor lock-in and compatibility issues
- **Static Tool Registration**: Tools must be explicitly defined and registered with the LLM at initialization time, limiting dynamic capability expansion
- **Tight Coupling**: Tool integrations are often tightly coupled with specific applications and LLM implementations, reducing reusability
- **Limited Discovery**: The LLM receives a fixed set of tool descriptions and must work within those constraints throughout the session
- **Direct Execution**: Function calls are processed directly by the host application's tool-handling logic without intermediate standardization

**MCP: Standardized Protocol Approach**
The Model Context Protocol transforms this paradigm through standardization and dynamic discovery:

**MCP Advantages:**
- **Open Standard**: Protocol specification is open and vendor-agnostic, enabling interoperability across different LLM providers and host applications
- **Dynamic Discovery**: MCP clients can query servers at runtime to discover available tools, resources, and prompts without requiring redeployment
- **Federated Architecture**: Enables composition of capabilities from multiple independent MCP servers into coherent workflows
- **Reusable Components**: MCP servers can be developed once and consumed by any compliant client, promoting code reuse and modularity
- **Standardized Communication**: All interactions follow the same protocol patterns, simplifying integration and debugging

**Architectural Comparison:**

**Function Calling Architecture:**
\`\`\`
LLM → Application Code → Tool Implementation → Response
      ↑ Proprietary Format                      ↓
      └─────────── Direct Response ──────────┘
\`\`\`

**MCP Architecture:**
\`\`\`
LLM → MCP Client → MCP Server → Tool Implementation → Response
      ↑ Standard   ↑ Standard  ↑ Implementation     ↓
      │ Protocol   │ Protocol  │ Specific           ↓
      └────────────┴──────────┴─────────────────────┘
\`\`\`

**When to Choose Each Approach:**

**Use Function Calling When:**
- Building simple applications with a fixed, small set of tools
- Working within a single LLM provider ecosystem
- Requiring minimal latency for tool invocation
- Developing prototypes or proof-of-concept applications
- Operating in environments where protocol standardization isn't a priority

**Use MCP When:**
- Building complex, multi-tool agent systems that need to scale
- Requiring interoperability between different LLM providers
- Developing enterprise applications that need to integrate with multiple existing systems
- Creating reusable tools that should work across different applications
- Building systems that need dynamic tool discovery and composition
- Working in environments where legacy system integration is crucial

**Migration Considerations:**
Organizations often begin with function calling for simplicity and migrate to MCP as complexity grows. The transition involves:
- **Tool Abstraction**: Converting direct function calls into MCP server implementations
- **Client Adaptation**: Updating applications to use MCP clients instead of direct tool handling
- **Capability Discovery**: Implementing dynamic tool discovery to replace static tool registration
- **Error Handling**: Adapting error handling to work with standardized MCP error formats

This architectural choice fundamentally impacts system scalability, maintainability, and interoperability, making MCP increasingly attractive for production enterprise deployments.`
    },
    {
      title: 'MCP Architecture and Component Interaction Patterns',
      content: `The Model Context Protocol operates through a sophisticated client-server architecture that standardizes how LLMs discover, access, and interact with external capabilities across diverse operational environments.

**Core Component Architecture**

**Large Language Model (LLM) - The Intelligence Core**
The LLM serves as the central intelligence unit that processes user requests, formulates execution plans, and determines when external capabilities are needed:
- **Request Analysis**: Parses user intent and identifies required external operations
- **Planning and Orchestration**: Develops multi-step workflows that may involve multiple MCP servers
- **Context Management**: Maintains conversation state and integrates results from external operations
- **Decision Making**: Determines optimal tool selection and parameter configuration for each operation

**MCP Client - The Protocol Intermediary**
The MCP client acts as the standardized bridge between the LLM and external systems:
- **Protocol Translation**: Converts LLM intent into properly formatted MCP requests
- **Server Discovery**: Identifies and catalogs available MCP servers and their capabilities  
- **Connection Management**: Establishes and maintains connections to multiple MCP servers
- **Result Processing**: Receives server responses and formats them for LLM consumption
- **Error Handling**: Manages connection failures, timeouts, and server-specific errors

**MCP Server - The Capability Gateway**
MCP servers expose external capabilities through standardized interfaces:
- **Capability Exposure**: Defines and publishes available tools, resources, and prompts
- **Request Processing**: Receives and validates incoming MCP requests from clients
- **Authentication & Authorization**: Manages access control and security policies
- **External System Integration**: Interfaces with underlying services, databases, and APIs
- **Response Formatting**: Standardizes response format for consistent client consumption

**Third-Party Services - The External Ecosystem**
These represent the actual systems and services that perform the requested operations:
- **Business Systems**: ERP, CRM, inventory management, and other enterprise applications
- **Data Sources**: Databases, data warehouses, file systems, and cloud storage services
- **External APIs**: Web services, SaaS platforms, and third-party integrations
- **Hardware Systems**: IoT devices, industrial equipment, and physical infrastructure

**Interaction Flow Patterns**

**1. Discovery Phase - Capability Exploration**
\`\`\`
MCP Client → Server: "What capabilities do you offer?"
MCP Server → Client: {tools: [...], resources: [...], prompts: [...]}
Client → LLM: "Available capabilities: X, Y, Z"
\`\`\`

**2. Planning Phase - Workflow Design**
\`\`\`
User → LLM: "Complex multi-step request"
LLM → Client: "I need tools A, B, C in sequence"
Client: Validates availability and dependencies
\`\`\`

**3. Execution Phase - Operation Coordination**
\`\`\`
LLM → Client: Tool invocation request with parameters
Client → Server: Standardized MCP request
Server → External System: Native API call
External System → Server: Raw response
Server → Client: Standardized MCP response  
Client → LLM: Formatted result for context integration
\`\`\`

**Transport Layer Mechanisms**

**Local Communication (JSON-RPC over STDIO)**
For same-machine deployments requiring high performance:
- **Inter-Process Communication**: Efficient data exchange between processes
- **Low Latency**: Minimal network overhead for rapid tool invocation
- **Security**: Leverages local system security boundaries
- **Resource Sharing**: Direct access to local file systems and processes

**Remote Communication (HTTP/Server-Sent Events)**
For distributed architectures and cloud deployments:
- **HTTP REST**: Standard web protocols for broad compatibility
- **Server-Sent Events (SSE)**: Real-time streaming for long-running operations
- **Load Balancing**: Distribute requests across multiple server instances
- **Network Security**: Standard web security protocols and practices

**Advanced Interaction Patterns**

**Batch Processing Workflows**
For large-scale data operations:
\`\`\`
LLM → Client: "Process 10,000 records"
Client → Server: Batch request with chunking strategy
Server: Processes in manageable chunks with progress reporting
Server → Client: Streaming progress updates via SSE
Client → LLM: Consolidated results with processing summary
\`\`\`

**Multi-Server Orchestration**
For complex workflows requiring multiple capabilities:
\`\`\`
LLM: Plans workflow requiring Database + FileSystem + Email servers
Client: Coordinates requests across three MCP servers
Server A (DB): Queries customer data
Server B (FS): Generates report file  
Server C (Email): Sends report to stakeholders
Client: Aggregates results and provides unified response
\`\`\`

**Error Recovery and Resilience**
Robust error handling across the MCP architecture:
- **Connection Failures**: Automatic retry with exponential backoff
- **Server Unavailability**: Graceful degradation with alternative tool suggestions
- **Partial Failures**: Transaction-like semantics for multi-step operations
- **Timeout Management**: Configurable timeouts with progress reporting

**Security and Access Control**
Comprehensive security model throughout the architecture:
- **Authentication**: Token-based or certificate-based client authentication
- **Authorization**: Fine-grained permissions for specific tools and resources
- **Audit Logging**: Complete request/response logging for security monitoring  
- **Data Encryption**: Transport-layer security for sensitive operations

This architectural approach enables sophisticated AI systems that can seamlessly integrate with complex enterprise environments while maintaining security, scalability, and reliability standards.`
    },
    {
      title: 'ADK Integration and Implementation Patterns with MCP',
      content: `Google's Agent Development Kit (ADK) provides comprehensive support for Model Context Protocol integration, offering built-in toolsets and streamlined development patterns for creating sophisticated agent-external system integrations.

**ADK MCPToolset Architecture**

The ADK's MCPToolset provides a high-level abstraction that simplifies MCP integration:

**Core MCPToolset Features:**
- **Connection Management**: Handles both local STDIO and remote HTTP server connections
- **Automatic Discovery**: Dynamically discovers and registers available tools from MCP servers
- **Tool Filtering**: Selective exposure of specific tools to maintain focused agent capabilities
- **Error Handling**: Robust error management with graceful degradation strategies
- **Security Integration**: Built-in authentication and authorization support

**Connection Parameter Types**

**StdioServerParameters - Local Server Integration**
For high-performance local integrations:
\`\`\`python
connection_params = StdioServerParameters(
    command='npx',  # Package runner command
    args=[
        '-y',  # Auto-confirm package installation
        '@modelcontextprotocol/server-filesystem',  # MCP server package
        '/absolute/path/to/managed/directory'  # Server-specific parameters
    ],
    env={  # Optional environment variables
        'SERVICE_ACCOUNT_PATH': '/path/to/credentials.json',
        'API_KEY': 'your_api_key_here'
    }
)
\`\`\`

**HttpServerParameters - Remote Server Integration**
For distributed architectures and cloud deployments:
\`\`\`python
connection_params = HttpServerParameters(
    url='http://localhost:8000',  # Server endpoint URL
    headers={  # Optional authentication headers
        'Authorization': 'Bearer your_token_here',
        'X-API-Version': '2024-01'
    },
    timeout=30  # Request timeout in seconds
)
\`\`\`

**Comprehensive Filesystem Integration Example**

**Directory Structure Setup:**
\`\`\`
project/
├── adk_agent_samples/
│   └── mcp_agent/
│       ├── __init__.py          # Package initialization
│       ├── agent.py             # Agent definition
│       └── mcp_managed_files/   # Managed file directory
│           ├── sample.txt
│           ├── data.json
│           └── readme.md
\`\`\`

**Complete Agent Implementation:**
\`\`\`python
import os
from google.adk.agents import LlmAgent
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, StdioServerParameters

# Dynamic path resolution for portability
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TARGET_FOLDER_PATH = os.path.join(SCRIPT_DIR, "mcp_managed_files")

# Ensure managed directory exists
os.makedirs(TARGET_FOLDER_PATH, exist_ok=True)

filesystem_agent = LlmAgent(
    model='gemini-2.0-flash',
    name='advanced_filesystem_assistant',
    instruction=f'''You are an advanced filesystem management assistant with comprehensive MCP capabilities.

    **Your Environment:**
    - Managed Directory: {TARGET_FOLDER_PATH}
    - Available Operations: list_directory, read_file, write_file
    - Security: Operations restricted to managed directory tree

    **Operational Guidelines:**
    1. Always verify directory contents before operations
    2. Provide detailed feedback on all file operations
    3. Handle errors gracefully with clear explanations
    4. Suggest alternatives when operations fail
    5. Maintain file organization and cleanliness

    **Advanced Capabilities:**
    - Analyze file contents for insights and patterns
    - Generate reports from multiple file sources
    - Maintain file metadata and organization
    - Coordinate complex multi-file operations
    ''',
    tools=[
        MCPToolset(
            connection_params=StdioServerParameters(
                command='npx',
                args=[
                    '-y',
                    '@modelcontextprotocol/server-filesystem',
                    TARGET_FOLDER_PATH
                ]
            ),
            tool_filter=['list_directory', 'read_file', 'write_file']
        )
    ]
)
\`\`\`

**Database Integration Patterns**

**SQLite Database MCP Integration:**
\`\`\`python
# Automatic database creation and seeding
def setup_sample_database(db_path: str):
    import sqlite3
    
    if os.path.exists(db_path):
        return
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create normalized schema
    cursor.execute('''
        CREATE TABLE departments (
            id INTEGER PRIMARY KEY,
            name TEXT UNIQUE NOT NULL,
            budget INTEGER DEFAULT 0
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE employees (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            department_id INTEGER,
            salary INTEGER,
            hire_date DATE DEFAULT CURRENT_DATE,
            FOREIGN KEY (department_id) REFERENCES departments (id)
        )
    ''')
    
    # Insert sample data with relationships
    departments_data = [
        ('Engineering', 1000000),
        ('Marketing', 500000),
        ('Sales', 750000)
    ]
    cursor.executemany('INSERT INTO departments (name, budget) VALUES (?, ?)', departments_data)
    
    employees_data = [
        ('Alice Johnson', 'alice@company.com', 1, 120000),
        ('Bob Smith', 'bob@company.com', 1, 100000),
        ('Carol Davis', 'carol@company.com', 2, 85000),
        ('David Wilson', 'david@company.com', 3, 90000)
    ]
    cursor.executemany('INSERT INTO employees (name, email, department_id, salary) VALUES (?, ?, ?, ?)', employees_data)
    
    conn.commit()
    conn.close()

# Database-enabled agent
DB_PATH = os.path.join(TARGET_FOLDER_PATH, "company.db")
setup_sample_database(DB_PATH)

database_agent = LlmAgent(
    model='gemini-2.0-flash',
    name='database_analyst_agent',
    instruction='''You are a database analyst with comprehensive query capabilities.
    
    **Available Database Schema:**
    - departments: id, name, budget
    - employees: id, name, email, department_id, salary, hire_date
    
    **Analysis Capabilities:**
    - Complex SQL queries with JOINs and aggregations
    - Department budget analysis and utilization
    - Employee salary distribution and statistics
    - Workforce demographics and reporting
    - Data integrity validation and monitoring
    ''',
    tools=[
        MCPToolset(
            connection_params=StdioServerParameters(
                command='npx',
                args=['-y', '@modelcontextprotocol/server-sqlite', DB_PATH]
            ),
            tool_filter=['query', 'execute', 'list_tables', 'describe_table']
        )
    ]
)
\`\`\`

**Advanced Multi-Server Integration**

**Enterprise Integration Pattern:**
\`\`\`python
from google.adk.agents import LlmAgent
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, StdioServerParameters, HttpServerParameters

enterprise_agent = LlmAgent(
    model='gemini-2.0-flash',
    name='enterprise_integration_agent',
    instruction='''You are an enterprise integration specialist with access to multiple systems.
    
    **Available Systems:**
    1. Filesystem: Document and file management
    2. Database: Employee and department data
    3. Web Services: External API integration
    4. Custom Tools: Business-specific operations
    
    **Integration Capabilities:**
    - Cross-system data correlation and analysis
    - Automated report generation from multiple sources
    - Workflow orchestration across systems
    - Data synchronization and validation
    - Security-aware operations with audit logging
    ''',
    tools=[
        # Filesystem integration
        MCPToolset(
            connection_params=StdioServerParameters(
                command='npx',
                args=['-y', '@modelcontextprotocol/server-filesystem', TARGET_FOLDER_PATH]
            ),
            tool_filter=['list_directory', 'read_file', 'write_file']
        ),
        
        # Database integration
        MCPToolset(
            connection_params=StdioServerParameters(
                command='npx', 
                args=['-y', '@modelcontextprotocol/server-sqlite', DB_PATH]
            ),
            tool_filter=['query', 'list_tables']
        ),
        
        # Custom HTTP service integration
        MCPToolset(
            connection_params=HttpServerParameters(
                url='http://localhost:8000'
            ),
            tool_filter=['calculate', 'analyze_text', 'greet']
        )
    ]
)
\`\`\`

**Development and Testing Workflow**

**1. Agent Development Setup:**
\`\`\`bash
# Navigate to agent samples directory
cd ./adk_agent_samples

# Start ADK Web interface for testing
adk web
\`\`\`

**2. Interactive Testing Commands:**
- "Show me all files in the managed directory"
- "Read the contents of sample.txt and analyze it"
- "Query the database for all employees with their department information"
- "Create a summary report combining file data and database information"
- "Use the custom calculator to analyze salary statistics"

**3. Production Deployment Considerations:**
- **Environment Configuration**: Proper environment variable management for credentials
- **Security Hardening**: Restricted file system access and database permissions
- **Performance Optimization**: Connection pooling and request batching
- **Monitoring Integration**: Comprehensive logging and health check endpoints
- **Scalability Planning**: Multi-instance deployment with load balancing

**Error Handling and Resilience Patterns**

\`\`\`python
# Enhanced error handling in agent instructions
error_handling_guidance = '''
**Error Handling Protocols:**
1. Server Connection Failures:
   - Attempt reconnection with exponential backoff
   - Provide clear error messages to users
   - Suggest alternative approaches when possible

2. Tool Execution Failures:
   - Validate parameters before sending requests
   - Parse error responses for actionable feedback
   - Maintain operation logs for debugging

3. Data Consistency Issues:
   - Verify data integrity after write operations
   - Provide rollback suggestions for failed transactions
   - Alert users to potential data conflicts

4. Performance Degradation:
   - Monitor response times and suggest optimizations
   - Implement request batching for large operations
   - Cache frequently accessed data when appropriate
'''
\`\`\`

This comprehensive ADK integration approach enables the development of robust, scalable agents that can seamlessly interact with complex enterprise environments while maintaining high standards of reliability, security, and performance.`
    },
    {
      title: 'FastMCP Server Development and Advanced Integration Patterns',
      content: `FastMCP provides a powerful Python framework for rapidly developing sophisticated MCP servers with minimal boilerplate code, enabling developers to focus on business logic while automatically handling protocol complexities and schema generation.

**FastMCP Core Architecture and Benefits**

**Key Framework Advantages:**
- **Decorator-Based Development**: Simple @tool, @resource, and @prompt decorators for capability registration
- **Automatic Schema Generation**: Intelligent interpretation of Python function signatures, type hints, and docstrings
- **Built-in Validation**: Automatic parameter validation and type checking based on function signatures
- **Multiple Transport Support**: HTTP, WebSocket, and STDIO transport mechanisms
- **Development Productivity**: Rapid prototyping with minimal configuration requirements

**Advanced Server Implementation Patterns**

**Comprehensive Business Logic Server:**
\`\`\`python
from fastmcp import FastMCP, Resource, Tool, Prompt
from typing import Dict, List, Any, Optional
import asyncio
import json
from datetime import datetime, timedelta
import hashlib

class AdvancedBusinessServer:
    """
    Sophisticated FastMCP server demonstrating advanced patterns:
    - State management and persistence
    - Complex validation and business rules
    - Multi-step workflow orchestration
    - Integration with external services
    """
    
    def __init__(self):
        self.server = FastMCP()
        self.data_store = {}
        self.audit_log = []
        self.session_cache = {}
        self.setup_tools()
        self.setup_resources()
        self.setup_prompts()
    
    def setup_tools(self):
        """Register comprehensive business tools."""
        
        @self.server.tool
        async def create_customer_profile(
            name: str,
            email: str,
            company: Optional[str] = None,
            tier: str = "standard"
        ) -> Dict[str, Any]:
            """
            Create comprehensive customer profile with validation and audit trail.
            
            Args:
                name: Customer full name (required)
                email: Valid email address (required)  
                company: Company/organization name (optional)
                tier: Customer tier (standard|premium|enterprise)
                
            Returns:
                Created customer profile with metadata
            """
            # Input validation
            if not name or not email:
                return {"error": "Name and email are required fields"}
            
            if tier not in ["standard", "premium", "enterprise"]:
                return {"error": "Invalid tier. Must be standard, premium, or enterprise"}
            
            if "@" not in email or "." not in email:
                return {"error": "Invalid email format"}
            
            # Check for duplicate email
            if any(customer.get("email") == email for customer in self.data_store.get("customers", {}).values()):
                return {"error": "Customer with this email already exists"}
            
            # Generate customer ID
            customer_id = hashlib.md5(f"{name}{email}{datetime.now()}".encode()).hexdigest()[:12]
            
            # Create customer profile
            profile = {
                "id": customer_id,
                "name": name,
                "email": email,
                "company": company,
                "tier": tier,
                "created_at": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat(),
                "status": "active",
                "preferences": {},
                "interaction_history": []
            }
            
            # Store customer
            if "customers" not in self.data_store:
                self.data_store["customers"] = {}
            
            self.data_store["customers"][customer_id] = profile
            
            # Audit trail
            self.audit_log.append({
                "action": "customer_created",
                "customer_id": customer_id,
                "timestamp": datetime.now().isoformat(),
                "details": {"name": name, "email": email, "tier": tier}
            })
            
            return {
                "success": True,
                "customer": profile,
                "message": f"Customer profile created successfully for {name}"
            }
        
        @self.server.tool
        async def analyze_customer_sentiment(
            customer_id: str,
            interaction_text: str,
            interaction_type: str = "support"
        ) -> Dict[str, Any]:
            """
            Analyze customer interaction sentiment and update customer profile.
            
            Args:
                customer_id: Unique customer identifier
                interaction_text: Text content of customer interaction
                interaction_type: Type of interaction (support|sales|feedback|complaint)
                
            Returns:
                Sentiment analysis results and updated customer context
            """
            # Validate customer exists
            if "customers" not in self.data_store or customer_id not in self.data_store["customers"]:
                return {"error": f"Customer {customer_id} not found"}
            
            customer = self.data_store["customers"][customer_id]
            
            # Simple sentiment analysis (in production, use proper NLP)
            positive_indicators = ["great", "excellent", "satisfied", "happy", "love", "perfect", "amazing"]
            negative_indicators = ["terrible", "awful", "disappointed", "frustrated", "hate", "worst", "horrible"]
            
            text_lower = interaction_text.lower()
            positive_count = sum(1 for word in positive_indicators if word in text_lower)
            negative_count = sum(1 for word in negative_indicators if word in text_lower)
            
            if positive_count > negative_count:
                sentiment = "positive"
                sentiment_score = 0.7 + (positive_count * 0.1)
            elif negative_count > positive_count:
                sentiment = "negative"
                sentiment_score = 0.3 - (negative_count * 0.1)
            else:
                sentiment = "neutral"
                sentiment_score = 0.5
            
            # Clamp sentiment score
            sentiment_score = max(0.0, min(1.0, sentiment_score))
            
            # Create interaction record
            interaction = {
                "timestamp": datetime.now().isoformat(),
                "type": interaction_type,
                "text": interaction_text,
                "sentiment": sentiment,
                "sentiment_score": sentiment_score,
                "analysis_metadata": {
                    "positive_indicators": positive_count,
                    "negative_indicators": negative_count,
                    "text_length": len(interaction_text)
                }
            }
            
            # Update customer profile
            customer["interaction_history"].append(interaction)
            customer["last_updated"] = datetime.now().isoformat()
            
            # Calculate overall sentiment trend
            recent_interactions = customer["interaction_history"][-5:]  # Last 5 interactions
            avg_sentiment = sum(i["sentiment_score"] for i in recent_interactions) / len(recent_interactions)
            
            if avg_sentiment >= 0.7:
                customer["sentiment_trend"] = "positive"
            elif avg_sentiment <= 0.3:
                customer["sentiment_trend"] = "negative"
            else:
                customer["sentiment_trend"] = "neutral"
            
            # Audit trail
            self.audit_log.append({
                "action": "sentiment_analyzed",
                "customer_id": customer_id,
                "timestamp": datetime.now().isoformat(),
                "details": {"sentiment": sentiment, "score": sentiment_score, "type": interaction_type}
            })
            
            return {
                "success": True,
                "analysis": {
                    "sentiment": sentiment,
                    "confidence_score": sentiment_score,
                    "interaction_type": interaction_type,
                    "customer_sentiment_trend": customer["sentiment_trend"],
                    "recommendations": self._generate_recommendations(customer, sentiment)
                },
                "customer_context": {
                    "name": customer["name"],
                    "tier": customer["tier"],
                    "total_interactions": len(customer["interaction_history"]),
                    "recent_sentiment_average": avg_sentiment
                }
            }
        
        @self.server.tool
        async def generate_business_report(
            report_type: str = "customer_summary",
            date_range_days: int = 30,
            include_details: bool = False
        ) -> Dict[str, Any]:
            """
            Generate comprehensive business intelligence reports from stored data.
            
            Args:
                report_type: Type of report (customer_summary|sentiment_analysis|audit_trail)
                date_range_days: Number of days to include in analysis
                include_details: Whether to include detailed breakdowns
                
            Returns:
                Formatted business report with insights and recommendations
            """
            cutoff_date = datetime.now() - timedelta(days=date_range_days)
            cutoff_iso = cutoff_date.isoformat()
            
            if report_type == "customer_summary":
                customers = self.data_store.get("customers", {})
                
                # Basic statistics
                total_customers = len(customers)
                tier_breakdown = {}
                sentiment_breakdown = {"positive": 0, "neutral": 0, "negative": 0}
                
                for customer in customers.values():
                    # Tier analysis
                    tier = customer.get("tier", "unknown")
                    tier_breakdown[tier] = tier_breakdown.get(tier, 0) + 1
                    
                    # Sentiment analysis
                    trend = customer.get("sentiment_trend", "neutral")
                    sentiment_breakdown[trend] += 1
                
                report = {
                    "report_type": "Customer Summary Report",
                    "generated_at": datetime.now().isoformat(),
                    "period": f"Last {date_range_days} days",
                    "summary": {
                        "total_customers": total_customers,
                        "tier_distribution": tier_breakdown,
                        "sentiment_distribution": sentiment_breakdown
                    },
                    "insights": [
                        f"Customer base consists of {total_customers} active customers",
                        f"Tier breakdown: {max(tier_breakdown, key=tier_breakdown.get)} tier is most common",
                        f"Sentiment health: {sentiment_breakdown['positive']} positive, {sentiment_breakdown['negative']} negative customers"
                    ]
                }
                
                if include_details:
                    report["detailed_customers"] = list(customers.values())
                
                return {"success": True, "report": report}
            
            elif report_type == "sentiment_analysis":
                # Analyze recent sentiment trends
                all_interactions = []
                for customer in self.data_store.get("customers", {}).values():
                    for interaction in customer.get("interaction_history", []):
                        if interaction["timestamp"] >= cutoff_iso:
                            all_interactions.append(interaction)
                
                if not all_interactions:
                    return {"success": True, "report": {"message": "No interactions in specified date range"}}
                
                # Sentiment analysis
                sentiment_counts = {"positive": 0, "neutral": 0, "negative": 0}
                type_breakdown = {}
                avg_sentiment_score = 0
                
                for interaction in all_interactions:
                    sentiment_counts[interaction["sentiment"]] += 1
                    interaction_type = interaction["type"]
                    type_breakdown[interaction_type] = type_breakdown.get(interaction_type, 0) + 1
                    avg_sentiment_score += interaction["sentiment_score"]
                
                avg_sentiment_score /= len(all_interactions)
                
                report = {
                    "report_type": "Sentiment Analysis Report", 
                    "generated_at": datetime.now().isoformat(),
                    "period": f"Last {date_range_days} days",
                    "analysis": {
                        "total_interactions": len(all_interactions),
                        "sentiment_breakdown": sentiment_counts,
                        "interaction_types": type_breakdown,
                        "average_sentiment_score": round(avg_sentiment_score, 3)
                    },
                    "insights": self._generate_sentiment_insights(sentiment_counts, avg_sentiment_score, type_breakdown)
                }
                
                return {"success": True, "report": report}
            
            else:
                return {"error": f"Unknown report type: {report_type}"}
    
    def _generate_recommendations(self, customer: Dict, sentiment: str) -> List[str]:
        """Generate contextual recommendations based on customer and sentiment."""
        recommendations = []
        
        if sentiment == "negative":
            recommendations.extend([
                "Consider immediate follow-up to address concerns",
                "Review interaction history for patterns",
                "Escalate to senior support if tier is premium/enterprise"
            ])
        elif sentiment == "positive":
            recommendations.extend([
                "Opportunity for upselling or tier upgrade",
                "Request testimonial or review",
                "Consider loyalty program enrollment"
            ])
        
        if customer.get("tier") == "enterprise":
            recommendations.append("Assign dedicated account manager")
        
        return recommendations
    
    def _generate_sentiment_insights(self, sentiment_counts, avg_score, type_breakdown):
        """Generate business insights from sentiment analysis."""
        insights = []
        
        total = sum(sentiment_counts.values())
        positive_pct = (sentiment_counts["positive"] / total) * 100
        negative_pct = (sentiment_counts["negative"] / total) * 100
        
        insights.append(f"Overall sentiment health: {positive_pct:.1f}% positive, {negative_pct:.1f}% negative")
        
        if avg_score >= 0.7:
            insights.append("Customer satisfaction is high - maintain current service quality")
        elif avg_score <= 0.3:
            insights.append("Customer satisfaction needs improvement - review support processes")
        
        most_common_type = max(type_breakdown, key=type_breakdown.get)
        insights.append(f"Most common interaction type: {most_common_type}")
        
        return insights
    
    def setup_resources(self):
        """Register business data resources."""
        
        @self.server.resource(uri="business://customer_database")
        def customer_database() -> str:
            """Provide access to customer database summary."""
            customers = self.data_store.get("customers", {})
            
            summary = {
                "total_customers": len(customers),
                "database_schema": {
                    "customer_fields": ["id", "name", "email", "company", "tier", "status", "created_at"],
                    "interaction_fields": ["timestamp", "type", "sentiment", "sentiment_score"],
                    "available_tiers": ["standard", "premium", "enterprise"]
                },
                "last_updated": datetime.now().isoformat()
            }
            
            return json.dumps(summary, indent=2)
        
        @self.server.resource(uri="business://audit_trail")
        def audit_trail() -> str:
            """Provide access to recent audit trail entries."""
            recent_entries = self.audit_log[-50:]  # Last 50 entries
            
            audit_summary = {
                "total_entries": len(self.audit_log),
                "recent_entries": recent_entries,
                "available_actions": list(set(entry["action"] for entry in self.audit_log))
            }
            
            return json.dumps(audit_summary, indent=2)
    
    def setup_prompts(self):
        """Register business workflow prompts."""
        
        @self.server.prompt(name="customer_onboarding_assistant")
        def customer_onboarding_prompt(customer_tier: str = "standard") -> str:
            """Generate customer onboarding workflow prompt."""
            
            tier_specific_guidance = {
                "standard": "Focus on core features and self-service resources",
                "premium": "Provide personalized setup assistance and training materials", 
                "enterprise": "Assign dedicated onboarding specialist and custom integration support"
            }
            
            guidance = tier_specific_guidance.get(customer_tier, tier_specific_guidance["standard"])
            
            return f'''You are a customer onboarding specialist for {customer_tier} tier customers.

**Onboarding Objectives:**
- Welcome new customer and confirm contact information
- Assess customer needs and use case requirements
- Configure initial account settings and preferences  
- Provide tier-appropriate training and documentation
- Schedule follow-up touchpoints for success monitoring

**Tier-Specific Approach:**
{guidance}

**Available Tools:**
- create_customer_profile: Set up complete customer record
- analyze_customer_sentiment: Monitor satisfaction during onboarding
- generate_business_report: Track onboarding success metrics

**Success Criteria:**
- Customer profile created with accurate information
- Initial positive sentiment interaction recorded
- Customer successfully completes first key workflow
- Follow-up schedule established based on tier requirements

Begin the onboarding process:'''
        
        @self.server.prompt(name="business_intelligence_analyst")
        def business_intelligence_prompt(focus_area: str = "comprehensive") -> str:
            """Generate business intelligence analysis prompt."""
            
            return f'''You are a business intelligence analyst specializing in customer data analysis.

**Analysis Focus: {focus_area}**

**Available Data Sources:**
- Customer profiles with tier and status information
- Interaction history with sentiment analysis
- Audit trail with operational metrics
- Business reports with trend analysis

**Analysis Capabilities:**
- Customer segmentation and tier analysis
- Sentiment trend identification and pattern recognition
- Operational efficiency metrics and recommendations
- Predictive insights for customer success and retention

**Reporting Standards:**
- Data-driven insights with supporting evidence
- Actionable recommendations for business improvement
- Clear visualizations of trends and patterns
- Executive summary with key findings

**Available Tools:**
- generate_business_report: Create comprehensive business reports
- analyze_customer_sentiment: Deep dive into customer satisfaction
- Access business resources for historical context

Begin your analysis:'''

# Server deployment and management
async def deploy_advanced_server():
    """Deploy the advanced FastMCP server with comprehensive capabilities."""
    
    server_instance = AdvancedBusinessServer()
    
    print("🚀 Deploying Advanced FastMCP Server")
    print("="*50)
    print("📊 Available Tools:")
    print("  • create_customer_profile - Customer management")
    print("  • analyze_customer_sentiment - Sentiment analysis")
    print("  • generate_business_report - Business intelligence")
    print("📁 Available Resources:")
    print("  • business://customer_database - Customer data access")
    print("  • business://audit_trail - Operational audit trail")
    print("💭 Available Prompts:")
    print("  • customer_onboarding_assistant - Guided onboarding")
    print("  • business_intelligence_analyst - Data analysis workflows")
    print("="*50)
    
    await server_instance.server.run(
        transport="http",
        host="127.0.0.1", 
        port=8000
    )

if __name__ == "__main__":
    asyncio.run(deploy_advanced_server())
\`\`\`

**Production Deployment Considerations**

**Scalability and Performance:**
- **Connection Pooling**: Manage database and external service connections efficiently
- **Caching Strategies**: Implement Redis or in-memory caching for frequently accessed data
- **Load Balancing**: Deploy multiple server instances behind a load balancer
- **Resource Monitoring**: Track CPU, memory, and I/O usage for optimization

**Security Hardening:**
- **Authentication**: Implement JWT or OAuth2 for secure client authentication
- **Authorization**: Role-based access control for sensitive operations
- **Input Validation**: Comprehensive sanitization of all input parameters
- **Audit Logging**: Complete request/response logging for security monitoring

**Integration Patterns:**
- **External API Integration**: Secure credential management and API key rotation
- **Database Integration**: Connection string encryption and credential vaulting
- **Message Queue Integration**: Async processing for long-running operations
- **Microservice Communication**: Service mesh integration for distributed architectures

This advanced FastMCP implementation demonstrates production-ready patterns for building sophisticated, scalable MCP servers that can integrate seamlessly with complex business environments while maintaining high standards of security, performance, and reliability.`
    }
  ],

  practicalExamples: [
    {
      title: 'Enterprise Document Management System with MCP',
      description: 'Comprehensive document management agent integrating filesystem operations, database metadata, and content analysis through standardized MCP interfaces',
      example: 'Corporate knowledge base system with intelligent document categorization, search, and cross-reference capabilities',
      steps: [
        'MCP Server Setup: Deploy filesystem MCP server with document directory access and database server for metadata storage',
        'Agent Configuration: Create ADK agent with MCPToolset connections to both filesystem and database servers with appropriate tool filtering',
        'Document Processing Pipeline: Implement workflows for document upload, content extraction, metadata generation, and database indexing',
        'Search and Retrieval: Enable natural language document search combining full-text content analysis with metadata querying',
        'Cross-Reference Analysis: Develop capabilities to identify relationships between documents and generate automated summaries',
        'Security Integration: Implement access control through MCP authentication and audit trail logging for all document operations'
      ]
    },
    {
      title: 'Multi-Modal Content Generation Pipeline with MCP',
      description: 'Creative content production system leveraging Google Cloud generative media services through MCP integration for coordinated multimedia creation',
      steps: [
        'Service Integration: Configure MCP connections to Google Cloud Imagen (images), Veo (video), Chirp 3 HD (voice), and Lyria (music) services',
        'Workflow Orchestration: Design agent capabilities to coordinate multi-step content creation workflows across different media types',
        'Creative Brief Processing: Enable natural language interpretation of creative requirements and automatic parameter translation for each service',
        'Asset Management: Integrate filesystem MCP server for generated asset storage, organization, and version control',
        'Quality Assurance: Implement content analysis and validation workflows to ensure generated media meets specifications',
        'Production Pipeline: Develop end-to-end workflows from concept to final delivery with automated packaging and distribution'
      ]
    },
    {
      title: 'Financial Analysis Platform with Real-Time Data Integration',
      description: 'Sophisticated financial analysis system combining market data APIs, calculation engines, and reporting through MCP standardization',
      example: 'Investment research platform with automated analysis, risk assessment, and portfolio optimization recommendations',
      steps: [
        'Data Source Integration: Deploy MCP servers for financial data APIs including market feeds, economic indicators, and company fundamentals',
        'Calculation Engine: Create FastMCP server with advanced financial calculation tools for risk metrics, valuation models, and portfolio analysis',
        'Database Integration: Configure MCP database connections for historical data storage, portfolio tracking, and audit trail maintenance',
        'Report Generation: Implement automated report creation combining real-time data analysis with historical trend analysis and visualization',
        'Risk Monitoring: Develop continuous monitoring workflows with alert generation for portfolio risk thresholds and market conditions',
        'Compliance Integration: Ensure all analysis and reporting meets regulatory requirements with comprehensive audit logging and data governance'
      ]
    }
  ],

  references: [
    'Model Context Protocol (MCP) Documentation: https://google.github.io/adk-docs/mcp/',
    'FastMCP Documentation and GitHub Repository: https://github.com/jlowin/fastmcp',
    'MCP Tools for Genmedia Services Documentation: https://google.github.io/adk-docs/mcp/#mcp-servers-for-google-cloud-genmedia',
    'MCP Toolbox for Databases: https://google.github.io/adk-docs/mcp/databases/',
    'Google Agent Development Kit (ADK) Official Documentation: https://google.github.io/adk-docs/',
    'Model Context Protocol Specification: https://spec.modelcontextprotocol.io/',
    'MCP Filesystem Server: https://github.com/modelcontextprotocol/servers/tree/main/src/filesystem'
  ],

  navigation: {
    previous: { href: '/chapters/learning-adaptation', title: 'Learning and Adaptation' },
    next: { href: '/chapters/goal-setting-monitoring', title: 'Goal Setting and Monitoring' }
  }
}
