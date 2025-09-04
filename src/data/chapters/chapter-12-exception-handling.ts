import { Chapter } from '../types'

export const exceptionHandlingChapter: Chapter = {
  id: 'exception-handling',
  number: 12,
  title: 'Exception Handling and Recovery',
  part: 'Part Three – Human-Centric Patterns',
  description: 'Build robust, resilient agents through comprehensive error detection, graceful failure handling, and intelligent recovery mechanisms that ensure reliable operation in unpredictable real-world environments.',
  readingTime: '30 min read',
  difficulty: 'Advanced',
  content: {
    overview: `Exception Handling and Recovery represents a critical foundation for building production-ready AI agents that can operate reliably in diverse, unpredictable real-world environments. Just as humans adapt to unexpected obstacles and recover from setbacks, intelligent agents require sophisticated systems to detect problems, implement appropriate responses, and restore stable operation when faced with various failures and anomalies.

This pattern addresses the fundamental challenge that AI agents operating in complex environments inevitably encounter unforeseen situations: tool failures, network issues, invalid data formats, service unavailability, and system malfunctions. Without structured exception management, agents become fragile systems prone to complete failure when encountering unexpected hurdles, limiting their deployment in critical applications where consistent performance is essential.

The pattern operates through three interconnected components: proactive error detection that identifies operational issues as they arise, comprehensive error handling strategies that implement appropriate responses to different failure types, and intelligent recovery mechanisms that restore agents to stable operational states. This systematic approach transforms brittle agents into robust, dependable systems capable of maintaining operational integrity, learning from failures, and functioning reliably in dynamic environments.

Effective implementation requires anticipation of potential failure modes, development of graduated response strategies, and establishment of recovery protocols that can handle everything from transient network issues to critical system failures. The pattern emphasizes both reactive handling of current errors and proactive preparation for likely failure scenarios, ensuring agents can continue providing value even when facing significant operational challenges.`,

    keyPoints: [
      'Provides systematic approach to error detection, handling, and recovery ensuring robust agent operation in unpredictable real-world environments',
      'Implements graduated response strategies including logging, retries, fallbacks, graceful degradation, and escalation based on error severity and type',
      'Enables proactive error detection through comprehensive monitoring of tool outputs, API responses, timeouts, and behavioral anomalies',
      'Supports intelligent recovery mechanisms including state rollback, self-correction, replanning, and human escalation for operational restoration',
      'Facilitates graceful degradation allowing agents to maintain partial functionality when complete recovery is not immediately possible',
      'Integrates comprehensive logging and diagnostic capabilities for error analysis, pattern recognition, and system improvement',
      'Enables multi-agent error handling patterns with specialized recovery agents and coordinated failure response strategies',
      'Essential for production deployment where operational reliability, user trust, and consistent performance are critical requirements'
    ],

    codeExample: `# Comprehensive Exception Handling and Recovery System
# Advanced implementation with multi-layered error detection, handling, and recovery

import asyncio
import logging
from typing import Dict, Any, List, Optional, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import traceback

# Google ADK imports for robust multi-agent error handling
from google.adk.agents import Agent, SequentialAgent, LlmAgent
from google.adk.tools import Tool, ToolResult

class ErrorSeverity(Enum):
    """Classification of error severity levels for appropriate response."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ErrorType(Enum):
    """Classification of error types for targeted handling strategies."""
    TOOL_FAILURE = "tool_failure"
    API_ERROR = "api_error"
    NETWORK_ISSUE = "network_issue"
    DATA_FORMAT_ERROR = "data_format_error"
    TIMEOUT = "timeout"
    AUTHENTICATION_ERROR = "authentication_error"
    RESOURCE_UNAVAILABLE = "resource_unavailable"
    VALIDATION_ERROR = "validation_error"
    SYSTEM_ERROR = "system_error"

@dataclass
class ErrorContext:
    """Comprehensive error context for detailed analysis and handling."""
    error_id: str
    error_type: ErrorType
    severity: ErrorSeverity
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    agent_id: str = ""
    tool_name: str = ""
    input_data: Dict[str, Any] = field(default_factory=dict)
    error_details: Dict[str, Any] = field(default_factory=dict)
    retry_count: int = 0
    max_retries: int = 3
    recovery_attempts: List[str] = field(default_factory=list)

@dataclass
class RecoveryStrategy:
    """Definition of recovery strategy with execution parameters."""
    name: str
    priority: int
    applicable_error_types: List[ErrorType]
    max_attempts: int
    delay_seconds: float
    success_criteria: Callable[[Any], bool]
    recovery_action: Callable[[ErrorContext], Any]

class RobustErrorHandler:
    """
    Comprehensive error handling system with multi-layered detection,
    graduated response strategies, and intelligent recovery mechanisms.
    """
    
    def __init__(self):
        """Initialize the robust error handling system."""
        self.error_history: List[ErrorContext] = []
        self.recovery_strategies: Dict[ErrorType, List[RecoveryStrategy]] = {}
        self.circuit_breakers: Dict[str, Dict] = {}
        self.monitoring_metrics: Dict[str, Any] = {}
        
        # Setup logging
        self.logger = self._setup_logging()
        
        # Initialize recovery strategies
        self._initialize_recovery_strategies()
        
        # Setup monitoring
        self._initialize_monitoring()
        
        print("🛡️ Robust Error Handling System initialized")
        print(f"📊 Recovery strategies: {sum(len(strategies) for strategies in self.recovery_strategies.values())}")
        print(f"🔍 Monitoring {len(self.monitoring_metrics)} metrics")
    
    def _setup_logging(self) -> logging.Logger:
        """Configure comprehensive logging for error tracking and analysis."""
        logger = logging.getLogger("RobustErrorHandler")
        logger.setLevel(logging.DEBUG)
        
        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s'
        )
        
        # Console handler for immediate feedback
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(detailed_formatter)
        
        # File handler for persistent logging
        file_handler = logging.FileHandler('agent_errors.log')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(detailed_formatter)
        
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
        
        return logger
    
    def _initialize_recovery_strategies(self):
        """Initialize comprehensive recovery strategies for different error types."""
        
        # Network/API retry strategy
        network_retry = RecoveryStrategy(
            name="exponential_backoff_retry",
            priority=1,
            applicable_error_types=[ErrorType.NETWORK_ISSUE, ErrorType.API_ERROR, ErrorType.TIMEOUT],
            max_attempts=5,
            delay_seconds=1.0,
            success_criteria=lambda result: result is not None and not isinstance(result, Exception),
            recovery_action=self._exponential_backoff_retry
        )
        
        # Fallback tool strategy
        tool_fallback = RecoveryStrategy(
            name="fallback_tool_usage",
            priority=2,
            applicable_error_types=[ErrorType.TOOL_FAILURE, ErrorType.API_ERROR],
            max_attempts=3,
            delay_seconds=0.5,
            success_criteria=lambda result: result is not None,
            recovery_action=self._try_fallback_tool
        )
        
        # Data validation and cleaning strategy
        data_recovery = RecoveryStrategy(
            name="data_validation_recovery",
            priority=1,
            applicable_error_types=[ErrorType.DATA_FORMAT_ERROR, ErrorType.VALIDATION_ERROR],
            max_attempts=2,
            delay_seconds=0.0,
            success_criteria=lambda result: self._validate_data_format(result),
            recovery_action=self._clean_and_validate_data
        )
        
        # Authentication refresh strategy
        auth_recovery = RecoveryStrategy(
            name="authentication_refresh",
            priority=1,
            applicable_error_types=[ErrorType.AUTHENTICATION_ERROR],
            max_attempts=2,
            delay_seconds=1.0,
            success_criteria=lambda result: result.get('authenticated', False) if isinstance(result, dict) else False,
            recovery_action=self._refresh_authentication
        )
        
        # Graceful degradation strategy
        graceful_degradation = RecoveryStrategy(
            name="graceful_degradation",
            priority=10,  # Lowest priority, last resort
            applicable_error_types=list(ErrorType),  # Applies to all error types
            max_attempts=1,
            delay_seconds=0.0,
            success_criteria=lambda result: result is not None,
            recovery_action=self._implement_graceful_degradation
        )
        
        # Populate recovery strategies dictionary
        for strategy in [network_retry, tool_fallback, data_recovery, auth_recovery, graceful_degradation]:
            for error_type in strategy.applicable_error_types:
                if error_type not in self.recovery_strategies:
                    self.recovery_strategies[error_type] = []
                self.recovery_strategies[error_type].append(strategy)
        
        # Sort strategies by priority
        for error_type in self.recovery_strategies:
            self.recovery_strategies[error_type].sort(key=lambda s: s.priority)
    
    def _initialize_monitoring(self):
        """Initialize comprehensive monitoring metrics and thresholds."""
        self.monitoring_metrics = {
            "total_errors": 0,
            "errors_by_type": {error_type.value: 0 for error_type in ErrorType},
            "errors_by_severity": {severity.value: 0 for severity in ErrorSeverity},
            "recovery_success_rate": 0.0,
            "average_recovery_time": 0.0,
            "circuit_breaker_trips": 0,
            "escalation_count": 0,
            "last_error_timestamp": None,
            "error_rate_per_hour": 0.0,
            "most_common_error_type": None,
            "agent_reliability_score": 1.0
        }
    
    def detect_error(self, 
                    result: Any, 
                    context: Dict[str, Any] = None,
                    agent_id: str = "",
                    tool_name: str = "") -> Optional[ErrorContext]:
        """
        Comprehensive error detection with multiple detection methods.
        
        Args:
            result: Result from tool execution or agent action
            context: Additional context information
            agent_id: ID of the agent that generated the result
            tool_name: Name of the tool that was executed
            
        Returns:
            ErrorContext if error detected, None otherwise
        """
        context = context or {}
        
        # Check for explicit exceptions
        if isinstance(result, Exception):
            return self._create_error_context(
                error_type=self._classify_exception(result),
                severity=self._determine_severity(result),
                message=str(result),
                agent_id=agent_id,
                tool_name=tool_name,
                input_data=context,
                error_details={"exception_type": type(result).__name__, "traceback": traceback.format_exc()}
            )
        
        # Check for API error patterns
        if isinstance(result, dict):
            # HTTP status code errors
            if "status_code" in result:
                status_code = result["status_code"]
                if 400 <= status_code < 600:
                    return self._create_error_context(
                        error_type=ErrorType.API_ERROR,
                        severity=ErrorSeverity.HIGH if status_code >= 500 else ErrorSeverity.MEDIUM,
                        message=f"API error with status code {status_code}",
                        agent_id=agent_id,
                        tool_name=tool_name,
                        input_data=context,
                        error_details={"status_code": status_code, "response": result}
                    )
            
            # Error field in response
            if "error" in result:
                return self._create_error_context(
                    error_type=ErrorType.API_ERROR,
                    severity=ErrorSeverity.MEDIUM,
                    message=result.get("error", "Unknown API error"),
                    agent_id=agent_id,
                    tool_name=tool_name,
                    input_data=context,
                    error_details=result
                )
        
        # Check for timeout indicators
        if isinstance(result, str) and "timeout" in result.lower():
            return self._create_error_context(
                error_type=ErrorType.TIMEOUT,
                severity=ErrorSeverity.MEDIUM,
                message="Operation timed out",
                agent_id=agent_id,
                tool_name=tool_name,
                input_data=context,
                error_details={"result": result}
            )
        
        # Check for data format issues
        if not self._validate_expected_format(result, context.get("expected_format")):
            return self._create_error_context(
                error_type=ErrorType.DATA_FORMAT_ERROR,
                severity=ErrorSeverity.LOW,
                message="Result does not match expected format",
                agent_id=agent_id,
                tool_name=tool_name,
                input_data=context,
                error_details={"result": str(result)[:200], "expected_format": context.get("expected_format")}
            )
        
        # Check circuit breaker status
        circuit_key = f"{agent_id}_{tool_name}"
        if self._is_circuit_breaker_open(circuit_key):
            return self._create_error_context(
                error_type=ErrorType.RESOURCE_UNAVAILABLE,
                severity=ErrorSeverity.HIGH,
                message=f"Circuit breaker open for {circuit_key}",
                agent_id=agent_id,
                tool_name=tool_name,
                input_data=context,
                error_details={"circuit_breaker_key": circuit_key}
            )
        
        return None
    
    async def handle_error(self, error_context: ErrorContext) -> Dict[str, Any]:
        """
        Comprehensive error handling with graduated response strategies.
        
        Args:
            error_context: Detailed error context
            
        Returns:
            Handling result with recovery status and actions taken
        """
        self.logger.error(f"Handling error: {error_context.error_id} - {error_context.message}")
        
        # Update monitoring metrics
        self._update_error_metrics(error_context)
        
        # Store in error history
        self.error_history.append(error_context)
        
        # Immediate logging
        await self._log_error_details(error_context)
        
        # Check for error pattern escalation
        if self._should_escalate_immediately(error_context):
            return await self._escalate_error(error_context)
        
        # Attempt recovery strategies
        recovery_result = await self._attempt_recovery(error_context)
        
        if recovery_result["success"]:
            self.logger.info(f"Successfully recovered from error {error_context.error_id}")
            return recovery_result
        
        # Implement fallback strategies if recovery failed
        fallback_result = await self._implement_fallback_strategies(error_context)
        
        if not fallback_result["success"]:
            # Final escalation if all recovery attempts failed
            self.logger.critical(f"All recovery attempts failed for error {error_context.error_id}")
            return await self._escalate_error(error_context)
        
        return fallback_result
    
    async def _attempt_recovery(self, error_context: ErrorContext) -> Dict[str, Any]:
        """
        Attempt recovery using appropriate strategies for the error type.
        
        Args:
            error_context: Error context with details
            
        Returns:
            Recovery attempt result
        """
        recovery_strategies = self.recovery_strategies.get(error_context.error_type, [])
        
        if not recovery_strategies:
            self.logger.warning(f"No recovery strategies available for error type {error_context.error_type}")
            return {"success": False, "reason": "No recovery strategies available"}
        
        for strategy in recovery_strategies:
            if error_context.retry_count >= strategy.max_attempts:
                continue
            
            self.logger.info(f"Attempting recovery strategy: {strategy.name}")
            
            try:
                # Apply delay if specified
                if strategy.delay_seconds > 0:
                    await asyncio.sleep(strategy.delay_seconds * (2 ** error_context.retry_count))  # Exponential backoff
                
                # Execute recovery action
                recovery_result = await strategy.recovery_action(error_context)
                
                # Validate success
                if strategy.success_criteria(recovery_result):
                    error_context.recovery_attempts.append(f"{strategy.name}: SUCCESS")
                    return {
                        "success": True,
                        "strategy": strategy.name,
                        "result": recovery_result,
                        "attempts": error_context.retry_count + 1
                    }
                else:
                    error_context.recovery_attempts.append(f"{strategy.name}: FAILED")
                    error_context.retry_count += 1
                    
            except Exception as e:
                self.logger.error(f"Recovery strategy {strategy.name} failed: {str(e)}")
                error_context.recovery_attempts.append(f"{strategy.name}: EXCEPTION - {str(e)}")
                error_context.retry_count += 1
        
        return {"success": False, "reason": "All recovery strategies failed"}
    
    async def _exponential_backoff_retry(self, error_context: ErrorContext) -> Any:
        """Exponential backoff retry strategy for transient failures."""
        self.logger.info(f"Attempting exponential backoff retry for {error_context.tool_name}")
        
        # Simulate retrying the original operation
        # In production, this would call the original tool/function
        await asyncio.sleep(0.1)  # Simulate network delay
        
        # Simulate success after retries
        if error_context.retry_count >= 2:  # Succeed after 2 retries
            return {"status": "success", "data": "Recovered through retry", "retry_count": error_context.retry_count}
        
        # Return failure to trigger another retry
        return None
    
    async def _try_fallback_tool(self, error_context: ErrorContext) -> Any:
        """Try alternative tool when primary tool fails."""
        self.logger.info(f"Attempting fallback tool for failed {error_context.tool_name}")
        
        # Map primary tools to fallback alternatives
        fallback_mapping = {
            "get_precise_location_info": "get_general_area_info",
            "high_accuracy_api": "basic_api",
            "premium_service": "standard_service"
        }
        
        fallback_tool = fallback_mapping.get(error_context.tool_name)
        if fallback_tool:
            self.logger.info(f"Using fallback tool: {fallback_tool}")
            return {
                "status": "success",
                "data": f"Result from {fallback_tool} (fallback)",
                "fallback_used": True,
                "original_tool": error_context.tool_name,
                "fallback_tool": fallback_tool
            }
        
        return None
    
    async def _clean_and_validate_data(self, error_context: ErrorContext) -> Any:
        """Attempt to clean and validate data format."""
        self.logger.info("Attempting data cleaning and validation")
        
        try:
            # Extract data from error context
            raw_data = error_context.input_data.get("data", "")
            
            # Basic data cleaning
            if isinstance(raw_data, str):
                cleaned_data = raw_data.strip().replace("\\n", " ").replace("\\t", " ")
                # Remove multiple spaces
                while "  " in cleaned_data:
                    cleaned_data = cleaned_data.replace("  ", " ")
                
                return {
                    "status": "success",
                    "data": cleaned_data,
                    "cleaned": True,
                    "original_length": len(raw_data),
                    "cleaned_length": len(cleaned_data)
                }
            
            return {"status": "success", "data": raw_data, "cleaned": False}
            
        except Exception as e:
            self.logger.error(f"Data cleaning failed: {str(e)}")
            return None
    
    async def _refresh_authentication(self, error_context: ErrorContext) -> Any:
        """Attempt to refresh authentication credentials."""
        self.logger.info("Attempting authentication refresh")
        
        # Simulate authentication refresh
        await asyncio.sleep(0.5)
        
        return {
            "status": "success",
            "authenticated": True,
            "token_refreshed": True,
            "timestamp": datetime.now().isoformat()
        }
    
    async def _implement_graceful_degradation(self, error_context: ErrorContext) -> Any:
        """Implement graceful degradation when full recovery is not possible."""
        self.logger.info(f"Implementing graceful degradation for {error_context.error_type}")
        
        degradation_strategies = {
            ErrorType.API_ERROR: "Using cached data or simplified functionality",
            ErrorType.NETWORK_ISSUE: "Operating in offline mode with limited features",
            ErrorType.RESOURCE_UNAVAILABLE: "Queuing request for later processing",
            ErrorType.TOOL_FAILURE: "Using alternative approach with reduced accuracy",
            ErrorType.TIMEOUT: "Providing partial results with timeout notification"
        }
        
        strategy = degradation_strategies.get(error_context.error_type, "Basic fallback response")
        
        return {
            "status": "degraded",
            "strategy": strategy,
            "full_functionality": False,
            "user_message": f"Service is operating in reduced mode: {strategy}",
            "error_context": error_context.error_id
        }
    
    async def _implement_fallback_strategies(self, error_context: ErrorContext) -> Dict[str, Any]:
        """Implement fallback strategies when primary recovery fails."""
        self.logger.info("Implementing fallback strategies")
        
        # Circuit breaker activation
        circuit_key = f"{error_context.agent_id}_{error_context.tool_name}"
        self._activate_circuit_breaker(circuit_key)
        
        # Graceful degradation as last resort
        degradation_result = await self._implement_graceful_degradation(error_context)
        
        return {
            "success": True,  # We provided some response, even if degraded
            "fallback": True,
            "degraded": True,
            "result": degradation_result,
            "circuit_breaker_activated": circuit_key
        }
    
    async def _escalate_error(self, error_context: ErrorContext) -> Dict[str, Any]:
        """Escalate error to human operators or higher-level systems."""
        self.logger.critical(f"Escalating error {error_context.error_id}")
        
        # Update monitoring metrics
        self.monitoring_metrics["escalation_count"] += 1
        
        # Create escalation message
        escalation_message = {
            "error_id": error_context.error_id,
            "severity": error_context.severity.value,
            "error_type": error_context.error_type.value,
            "message": error_context.message,
            "agent_id": error_context.agent_id,
            "tool_name": error_context.tool_name,
            "recovery_attempts": error_context.recovery_attempts,
            "timestamp": error_context.timestamp.isoformat(),
            "requires_human_intervention": True
        }
        
        # In production, this would send notifications to operators
        print(f"🚨 ESCALATION ALERT: {json.dumps(escalation_message, indent=2)}")
        
        return {
            "success": False,
            "escalated": True,
            "escalation_message": escalation_message,
            "user_message": "We've encountered an issue that requires attention. Our team has been notified."
        }
    
    def _create_error_context(self, 
                            error_type: ErrorType, 
                            severity: ErrorSeverity, 
                            message: str,
                            agent_id: str = "",
                            tool_name: str = "",
                            input_data: Dict[str, Any] = None,
                            error_details: Dict[str, Any] = None) -> ErrorContext:
        """Create comprehensive error context."""
        error_id = f"{error_type.value}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        return ErrorContext(
            error_id=error_id,
            error_type=error_type,
            severity=severity,
            message=message,
            agent_id=agent_id,
            tool_name=tool_name,
            input_data=input_data or {},
            error_details=error_details or {}
        )
    
    def _classify_exception(self, exception: Exception) -> ErrorType:
        """Classify exception type for appropriate handling."""
        exception_mappings = {
            ConnectionError: ErrorType.NETWORK_ISSUE,
            TimeoutError: ErrorType.TIMEOUT,
            ValueError: ErrorType.VALIDATION_ERROR,
            KeyError: ErrorType.DATA_FORMAT_ERROR,
            PermissionError: ErrorType.AUTHENTICATION_ERROR,
            FileNotFoundError: ErrorType.RESOURCE_UNAVAILABLE
        }
        
        return exception_mappings.get(type(exception), ErrorType.SYSTEM_ERROR)
    
    def _determine_severity(self, error: Union[Exception, Dict, str]) -> ErrorSeverity:
        """Determine error severity based on error characteristics."""
        if isinstance(error, Exception):
            critical_exceptions = [SystemExit, KeyboardInterrupt, MemoryError]
            if type(error) in critical_exceptions:
                return ErrorSeverity.CRITICAL
            
            high_exceptions = [ConnectionError, PermissionError]
            if type(error) in high_exceptions:
                return ErrorSeverity.HIGH
        
        if isinstance(error, dict) and "status_code" in error:
            status_code = error["status_code"]
            if status_code >= 500:
                return ErrorSeverity.HIGH
            elif status_code >= 400:
                return ErrorSeverity.MEDIUM
        
        return ErrorSeverity.LOW
    
    def _validate_expected_format(self, result: Any, expected_format: str = None) -> bool:
        """Validate result against expected format."""
        if expected_format is None:
            return True
        
        if expected_format == "json" and isinstance(result, dict):
            return True
        elif expected_format == "string" and isinstance(result, str):
            return True
        elif expected_format == "list" and isinstance(result, list):
            return True
        elif expected_format == "number" and isinstance(result, (int, float)):
            return True
        
        return False
    
    def _validate_data_format(self, data: Any) -> bool:
        """Validate data format for recovery success criteria."""
        return data is not None and not isinstance(data, Exception)
    
    def _should_escalate_immediately(self, error_context: ErrorContext) -> bool:
        """Determine if error should be escalated immediately."""
        immediate_escalation_conditions = [
            error_context.severity == ErrorSeverity.CRITICAL,
            error_context.error_type == ErrorType.AUTHENTICATION_ERROR and error_context.retry_count > 2,
            "security" in error_context.message.lower(),
            "unauthorized" in error_context.message.lower()
        ]
        
        return any(immediate_escalation_conditions)
    
    def _is_circuit_breaker_open(self, circuit_key: str) -> bool:
        """Check if circuit breaker is open for given key."""
        if circuit_key not in self.circuit_breakers:
            return False
        
        circuit = self.circuit_breakers[circuit_key]
        return circuit.get("open", False) and datetime.now() < circuit.get("reset_time", datetime.now())
    
    def _activate_circuit_breaker(self, circuit_key: str, timeout_minutes: int = 5):
        """Activate circuit breaker for failing component."""
        self.circuit_breakers[circuit_key] = {
            "open": True,
            "activated_at": datetime.now(),
            "reset_time": datetime.now() + timedelta(minutes=timeout_minutes),
            "failure_count": self.circuit_breakers.get(circuit_key, {}).get("failure_count", 0) + 1
        }
        
        self.monitoring_metrics["circuit_breaker_trips"] += 1
        self.logger.warning(f"Circuit breaker activated for {circuit_key}")
    
    def _update_error_metrics(self, error_context: ErrorContext):
        """Update comprehensive monitoring metrics."""
        self.monitoring_metrics["total_errors"] += 1
        self.monitoring_metrics["errors_by_type"][error_context.error_type.value] += 1
        self.monitoring_metrics["errors_by_severity"][error_context.severity.value] += 1
        self.monitoring_metrics["last_error_timestamp"] = datetime.now()
        
        # Update most common error type
        error_counts = self.monitoring_metrics["errors_by_type"]
        self.monitoring_metrics["most_common_error_type"] = max(error_counts, key=error_counts.get)
        
        # Update reliability score
        total_operations = self.monitoring_metrics.get("total_operations", 1)
        success_rate = 1.0 - (self.monitoring_metrics["total_errors"] / total_operations)
        self.monitoring_metrics["agent_reliability_score"] = max(0.0, success_rate)
    
    async def _log_error_details(self, error_context: ErrorContext):
        """Comprehensive error logging with structured data."""
        log_entry = {
            "error_id": error_context.error_id,
            "timestamp": error_context.timestamp.isoformat(),
            "error_type": error_context.error_type.value,
            "severity": error_context.severity.value,
            "message": error_context.message,
            "agent_id": error_context.agent_id,
            "tool_name": error_context.tool_name,
            "retry_count": error_context.retry_count,
            "input_data": error_context.input_data,
            "error_details": error_context.error_details
        }
        
        self.logger.error(f"ERROR_DETAILS: {json.dumps(log_entry, indent=2)}")
    
    def get_monitoring_report(self) -> Dict[str, Any]:
        """Generate comprehensive monitoring and health report."""
        return {
            "timestamp": datetime.now().isoformat(),
            "system_health": {
                "reliability_score": self.monitoring_metrics["agent_reliability_score"],
                "total_errors": self.monitoring_metrics["total_errors"],
                "escalation_rate": self.monitoring_metrics["escalation_count"] / max(1, self.monitoring_metrics["total_errors"]),
                "circuit_breaker_trips": self.monitoring_metrics["circuit_breaker_trips"]
            },
            "error_patterns": {
                "by_type": self.monitoring_metrics["errors_by_type"],
                "by_severity": self.monitoring_metrics["errors_by_severity"],
                "most_common": self.monitoring_metrics["most_common_error_type"]
            },
            "recent_errors": [
                {
                    "id": error.error_id,
                    "type": error.error_type.value,
                    "severity": error.severity.value,
                    "timestamp": error.timestamp.isoformat(),
                    "recovered": len(error.recovery_attempts) > 0
                }
                for error in self.error_history[-10:]  # Last 10 errors
            ],
            "active_circuit_breakers": {
                key: {
                    "failure_count": circuit["failure_count"],
                    "activated_at": circuit["activated_at"].isoformat(),
                    "reset_time": circuit["reset_time"].isoformat()
                }
                for key, circuit in self.circuit_breakers.items()
                if circuit.get("open", False)
            }
        }

# ADK Implementation: Robust Location Retrieval with Exception Handling
class RobustLocationAgent:
    """
    Comprehensive location retrieval system demonstrating advanced exception
    handling patterns with ADK SequentialAgent and specialized recovery agents.
    """
    
    def __init__(self):
        """Initialize the robust location agent system."""
        self.error_handler = RobustErrorHandler()
        self.setup_location_tools()
        self.create_agent_pipeline()
        
        print("🗺️ Robust Location Agent initialized")
        print("🔗 Multi-agent pipeline with exception handling ready")
    
    def setup_location_tools(self):
        """Setup location tools with error simulation for demonstration."""
        
        async def get_precise_location_info(address: str) -> Dict[str, Any]:
            """Primary location tool that may fail."""
            # Simulate various failure scenarios
            import random
            
            failure_scenarios = [
                {"error": "API rate limit exceeded", "status_code": 429},
                {"error": "Invalid address format", "status_code": 400},
                {"error": "Service temporarily unavailable", "status_code": 503},
                None  # Success case
            ]
            
            # Simulate failure 60% of the time for demonstration
            if random.random() < 0.6:
                failure = random.choice(failure_scenarios[:-1])
                if failure:
                    raise Exception(f"Location API Error: {failure['error']}")
            
            return {
                "address": address,
                "coordinates": {"lat": 37.7749, "lng": -122.4194},
                "confidence": "high",
                "source": "precise_api"
            }
        
        async def get_general_area_info(city: str) -> Dict[str, Any]:
            """Fallback location tool with broader area information."""
            # Simulate higher success rate for fallback
            import random
            
            if random.random() < 0.9:  # 90% success rate
                return {
                    "city": city,
                    "region": "California",
                    "country": "United States",
                    "coordinates": {"lat": 37.7749, "lng": -122.4194},
                    "confidence": "medium",
                    "source": "general_api"
                }
            
            raise Exception("General location service unavailable")
        
        self.location_tools = {
            "get_precise_location_info": get_precise_location_info,
            "get_general_area_info": get_general_area_info
        }
    
    def create_agent_pipeline(self):
        """Create ADK SequentialAgent pipeline with exception handling."""
        
        # Agent 1: Primary location handler with error detection
        primary_handler = Agent(
            name="primary_location_handler",
            model="gemini-2.0-flash-exp",
            instruction="""
You are a precise location lookup specialist.

Your task:
1. Use the get_precise_location_info tool with the provided address
2. If successful, store the result in state["location_result"]
3. If the tool fails or returns an error, set state["primary_location_failed"] = True
4. Always provide a clear status of your attempt

Handle any errors gracefully and provide informative feedback.
            """,
            tools=[self.location_tools["get_precise_location_info"]]
        )
        
        # Agent 2: Fallback handler with error recovery
        fallback_handler = Agent(
            name="fallback_location_handler", 
            model="gemini-2.0-flash-exp",
            instruction="""
You are a fallback location handler that provides recovery when precise lookup fails.

Your task:
1. Check if state["primary_location_failed"] is True
2. If True:
   - Extract the city name from the user's original query
   - Use the get_general_area_info tool with the city name
   - Store the result in state["location_result"]
   - Set state["fallback_used"] = True
3. If False, do nothing (primary handler succeeded)

Provide helpful messages about using fallback data when precision isn't available.
            """,
            tools=[self.location_tools["get_general_area_info"]]
        )
        
        # Agent 3: Error analysis and response coordination
        error_analysis_agent = Agent(
            name="error_analysis_agent",
            model="gemini-2.0-flash-exp", 
            instruction="""
You are an error analysis and response coordination specialist.

Your task:
1. Analyze the current state for any errors or failures
2. If both primary and fallback failed, implement graceful degradation:
   - Acknowledge the limitation
   - Provide alternative suggestions
   - Set state["graceful_degradation"] = True
3. Log error patterns for system improvement
4. Prepare appropriate user communication

Focus on maintaining user experience even during system failures.
            """,
            tools=[]
        )
        
        # Agent 4: Response synthesis and user communication
        response_agent = Agent(
            name="response_synthesis_agent",
            model="gemini-2.0-flash-exp",
            instruction="""
You are a response synthesis specialist responsible for final user communication.

Your task:
1. Review all location information in state["location_result"]
2. Check for any fallback usage or graceful degradation flags
3. Present the information clearly and transparently to the user
4. Include appropriate disclaimers for fallback or degraded responses
5. If no location data available, provide helpful alternatives

Always be honest about data quality and limitations while remaining helpful.
            """,
            tools=[]
        )
        
        # Create the robust sequential agent pipeline
        self.robust_location_agent = SequentialAgent(
            name="robust_location_retrieval_system",
            sub_agents=[primary_handler, fallback_handler, error_analysis_agent, response_agent]
        )
    
    async def process_location_request(self, address: str) -> Dict[str, Any]:
        """
        Process location request through the robust pipeline with comprehensive error handling.
        
        Args:
            address: Address to look up
            
        Returns:
            Processing result with location data and error handling status
        """
        print(f"\\n🔍 Processing location request: {address}")
        print("="*60)
        
        try:
            # Execute the robust agent pipeline
            result = await self.robust_location_agent.run(
                input_text=f"Find location information for: {address}",
                state={"user_query": address}
            )
            
            # Analyze the final state for error handling metrics
            final_state = result.get("state", {})
            
            processing_summary = {
                "request": address,
                "success": "location_result" in final_state,
                "primary_failed": final_state.get("primary_location_failed", False),
                "fallback_used": final_state.get("fallback_used", False),
                "graceful_degradation": final_state.get("graceful_degradation", False),
                "location_data": final_state.get("location_result"),
                "response": result.get("output", ""),
                "processing_time": datetime.now().isoformat(),
                "error_handling_effective": True
            }
            
            # Log success metrics
            self.error_handler.monitoring_metrics["total_operations"] = \
                self.error_handler.monitoring_metrics.get("total_operations", 0) + 1
            
            print(f"✅ Location request processed successfully")
            if processing_summary["fallback_used"]:
                print(f"⚠️ Fallback strategy was used")
            if processing_summary["graceful_degradation"]:
                print(f"🔄 Graceful degradation implemented")
            
            return processing_summary
            
        except Exception as e:
            # Handle pipeline-level errors
            error_context = self.error_handler.detect_error(
                result=e,
                context={"address": address, "pipeline": "location_retrieval"},
                agent_id="robust_location_agent"
            )
            
            if error_context:
                recovery_result = await self.error_handler.handle_error(error_context)
                return {
                    "request": address,
                    "success": False,
                    "error_handled": True,
                    "error_context": error_context.error_id,
                    "recovery_result": recovery_result,
                    "user_message": "We encountered an issue processing your location request. Please try again or contact support."
                }
            
            return {
                "request": address,
                "success": False,
                "error": str(e),
                "user_message": "An unexpected error occurred. Please try again."
            }

# Demonstration and Usage Examples
async def demonstrate_exception_handling():
    """
    Comprehensive demonstration of exception handling and recovery patterns
    with real-world scenarios and failure simulation.
    """
    
    print("🛡️ EXCEPTION HANDLING AND RECOVERY DEMONSTRATION")
    print("="*70)
    
    # Initialize robust location agent
    location_agent = RobustLocationAgent()
    
    # Test scenarios with various failure patterns
    test_addresses = [
        "1600 Amphitheatre Parkway, Mountain View, CA",  # May succeed or fail
        "Invalid Address Format",                         # Likely to fail primary, succeed fallback
        "123 Main Street, Anytown, USA",                 # Testing fallback scenarios
        "San Francisco, CA",                             # City-level query
        "Nonexistent Location, Mars"                     # Complete failure scenario
    ]
    
    results = []
    
    for i, address in enumerate(test_addresses, 1):
        print(f"\\n{'='*20} Test {i}/{len(test_addresses)} {'='*20}")
        result = await location_agent.process_location_request(address)
        results.append(result)
        
        # Brief pause between requests
        await asyncio.sleep(1)
    
    # Generate comprehensive monitoring report
    monitoring_report = location_agent.error_handler.get_monitoring_report()
    
    print(f"\\n{'='*25} 📊 FINAL REPORT {'='*25}")
    print(f"Total Requests: {len(test_addresses)}")
    print(f"Success Rate: {sum(1 for r in results if r['success']) / len(results):.1%}")
    print(f"Fallback Usage: {sum(1 for r in results if r.get('fallback_used', False))}")
    print(f"Graceful Degradation: {sum(1 for r in results if r.get('graceful_degradation', False))}")
    print(f"System Reliability: {monitoring_report['system_health']['reliability_score']:.3f}")
    print(f"Error Types: {monitoring_report['error_patterns']['by_type']}")
    print("="*68)
    
    return results, monitoring_report

# Main execution example
if __name__ == "__main__":
    async def main():
        # Run comprehensive demonstration
        results, report = await demonstrate_exception_handling()
        
        print("\\n🎯 EXCEPTION HANDLING DEMONSTRATION COMPLETE!")
        print(f"Processed {len(results)} requests with comprehensive error handling")
        print("System demonstrated resilience through failures and recoveries")
    
    asyncio.run(main())`,

    practicalApplications: [
      '🤖 Customer Service Chatbots: Detect database connectivity issues, implement retry mechanisms, use cached responses as fallback, gracefully degrade to basic FAQ responses, and escalate complex queries to human agents when automated systems fail',
      '💹 Automated Trading Systems: Handle "insufficient funds" and "market closed" errors through validation, implement circuit breakers for repeated failures, log all trading errors for analysis, and notify users of execution issues without exposing system vulnerabilities',
      '🏠 Smart Home Automation: Detect device communication failures, retry commands with exponential backoff, switch to backup devices when available, notify users of device status changes, and maintain partial functionality during network outages',
      '📊 Data Processing Agents: Skip corrupted files while continuing batch processing, validate data formats before processing, implement checkpoint systems for recovery, log processing errors with context, and provide detailed completion reports including failed items',
      '🕷️ Web Scraping Systems: Handle CAPTCHA challenges through proxy rotation, adapt to website structure changes with flexible parsing, implement rate limiting for respectful scraping, retry failed requests with delays, and maintain detailed failure logs for optimization',
      '🤖 Robotics and Manufacturing: Detect sensor feedback anomalies, implement safety stops for critical failures, retry precision operations with adjustments, escalate persistent hardware issues to maintenance teams, and maintain production logs for quality assurance',
      '📱 Mobile App Agents: Handle network connectivity issues gracefully, implement offline mode capabilities, sync data when connectivity returns, provide user feedback during service disruptions, and maintain app functionality during partial system failures',
      '🔒 Security Monitoring: Detect authentication failures and potential security breaches, implement account lockout mechanisms for repeated failures, escalate security incidents immediately, log all security events comprehensively, and maintain system integrity during attacks'
    ],

    nextSteps: [
      'Implement comprehensive error detection systems that monitor tool outputs, API responses, network connectivity, and behavioral anomalies in real-time',
      'Design graduated response strategies with logging, exponential backoff retries, fallback mechanisms, graceful degradation, and escalation protocols based on error severity',
      'Create intelligent recovery mechanisms including state rollback, self-correction through replanning, authentication refresh, and circuit breaker patterns for failing components',
      'Build robust monitoring and alerting systems that track error patterns, success rates, recovery effectiveness, and system health metrics with automated reporting',
      'Develop multi-agent error handling patterns with specialized recovery agents, coordinated failure response strategies, and distributed resilience mechanisms',
      'Establish comprehensive logging and diagnostic capabilities for error analysis, pattern recognition, root cause analysis, and continuous system improvement',
      'Implement user-centric error communication that provides clear, helpful messages while maintaining system security and avoiding technical jargon',
      'Design testing frameworks for error handling validation including failure injection, recovery testing, load testing, and resilience verification across different failure scenarios'
    ]
  },

  sections: [
    {
      title: 'Error Detection and Classification Patterns in Agentic Systems',
      content: `Effective error detection forms the foundation of robust exception handling, requiring sophisticated monitoring systems that can identify various types of failures across different operational layers and classify them appropriately for targeted response strategies.

**Multi-Layer Error Detection Architecture**

**Application Layer Detection**
At the highest level, agents must monitor their own operational outputs and behavior:
- **Tool Output Validation**: Systematic examination of tool results for expected format, data completeness, and logical consistency
- **Response Quality Assessment**: Evaluation of agent responses against quality metrics, coherence standards, and user expectations
- **Behavioral Anomaly Detection**: Identification of unusual patterns in agent decision-making, response times, or resource utilization
- **Goal Achievement Monitoring**: Tracking progress toward objectives and identifying when agents deviate from expected performance trajectories

**API and Service Layer Detection**
Agents interacting with external services require comprehensive API monitoring:
- **HTTP Status Code Analysis**: Systematic handling of 4xx client errors, 5xx server errors, and unexpected response codes
- **Response Format Validation**: Verification that API responses match expected schemas, data types, and structural requirements
- **Timeout Detection**: Monitoring for requests that exceed reasonable time limits and implementing appropriate timeout handling
- **Rate Limit Recognition**: Detection of API throttling and implementation of backoff strategies to prevent service blocking

**Network and Infrastructure Layer Detection**
Lower-level infrastructure monitoring ensures system reliability:
- **Connection Failure Detection**: Monitoring for network connectivity issues, DNS resolution failures, and connection timeout scenarios
- **Resource Availability Monitoring**: Tracking memory usage, CPU utilization, disk space, and other critical system resources
- **Service Dependency Health**: Continuous monitoring of external service availability and performance characteristics
- **Security Event Detection**: Identification of authentication failures, unauthorized access attempts, and potential security breaches

**Error Classification Framework**

**Severity-Based Classification**
Errors are classified by their potential impact on system operation:

**Critical Errors**: System-threatening failures requiring immediate escalation
- Security breaches or authentication compromises
- Data corruption or loss scenarios  
- System crash conditions or memory exhaustion
- Safety-critical failures in robotics or autonomous systems

**High Severity Errors**: Significant functional impact but system remains operational
- Primary API service unavailability
- Database connection failures affecting core functionality
- Authentication service disruptions
- Major tool or service component failures

**Medium Severity Errors**: Partial functionality loss with workarounds available
- Secondary service unavailability with fallback options
- Data format inconsistencies that can be corrected
- Temporary network connectivity issues
- Non-critical tool failures with alternatives

**Low Severity Errors**: Minor issues with minimal operational impact
- Cosmetic data formatting problems
- Optional feature unavailability
- Performance degradation within acceptable limits
- Non-essential tool timeouts

**Type-Based Classification**
Errors are categorized by their root cause for targeted handling:

**Transient Errors**: Temporary issues likely to resolve with retry
- Network connectivity hiccups
- Temporary service overload (HTTP 503)
- Rate limiting (HTTP 429)
- Transient database connection issues

**Configuration Errors**: Issues with system setup or parameters
- Invalid API keys or expired credentials
- Incorrect service endpoints or URLs
- Missing required configuration parameters
- Environment-specific setting mismatches

**Data Errors**: Issues with input data or processing
- Invalid input formats or schemas
- Missing required data fields
- Data type mismatches or conversion errors
- Corrupted or incomplete data sets

**Logic Errors**: Issues with agent reasoning or decision-making
- Invalid parameter combinations
- Logical inconsistencies in agent plans
- Constraint violations or rule conflicts
- Unexpected edge cases in reasoning logic

**Advanced Detection Techniques**

**Pattern Recognition Detection**
Machine learning approaches for sophisticated error identification:
- **Anomaly Detection Models**: Statistical models that identify unusual patterns in agent behavior, response times, or output characteristics
- **Trend Analysis**: Time-series analysis to identify degrading performance or emerging failure patterns before they become critical
- **Correlation Analysis**: Identification of relationships between different error types and environmental conditions
- **Predictive Failure Modeling**: Models that predict potential failures based on current system state and historical patterns

**Context-Aware Detection**
Error detection that considers operational context:
- **User Context**: Adapting error sensitivity based on user type, request criticality, and service level agreements
- **Environmental Context**: Considering time of day, system load, and external factors that might influence error likelihood
- **Historical Context**: Leveraging past error patterns and resolution success rates to inform current detection thresholds
- **Business Context**: Prioritizing error detection based on business impact and operational criticality

**Real-Time Monitoring Implementation**
\`\`\`python
class AdvancedErrorDetector:
    def __init__(self):
        self.detection_rules = []
        self.anomaly_baselines = {}
        self.pattern_history = []
        
    def detect_errors(self, operation_result, context):
        # Multi-layer detection approach
        detected_errors = []
        
        # Explicit error detection
        explicit_error = self.check_explicit_errors(operation_result)
        if explicit_error:
            detected_errors.append(explicit_error)
        
        # Pattern-based detection
        pattern_error = self.check_patterns(operation_result, context)
        if pattern_error:
            detected_errors.append(pattern_error)
        
        # Anomaly detection
        anomaly_error = self.check_anomalies(operation_result, context)
        if anomaly_error:
            detected_errors.append(anomaly_error)
        
        return detected_errors
    
    def check_explicit_errors(self, result):
        # Check for obvious error indicators
        if isinstance(result, Exception):
            return self.classify_exception(result)
        
        if isinstance(result, dict) and 'error' in result:
            return self.classify_api_error(result)
        
        return None
    
    def check_patterns(self, result, context):
        # Pattern-based error detection
        for rule in self.detection_rules:
            if rule.matches(result, context):
                return rule.create_error_context(result, context)
        
        return None
    
    def check_anomalies(self, result, context):
        # Statistical anomaly detection
        if self.is_anomalous(result, context):
            return self.create_anomaly_error(result, context)
        
        return None
\`\`\`

**Integration with Error Handling Pipeline**
Error detection systems integrate seamlessly with handling and recovery mechanisms:
- **Real-Time Alerting**: Immediate notification of detected errors to appropriate handling systems
- **Context Preservation**: Maintaining detailed context about detected errors for effective recovery strategies
- **Performance Impact Monitoring**: Ensuring detection systems don't introduce significant overhead to normal operations
- **Adaptive Thresholds**: Dynamic adjustment of detection sensitivity based on system performance and operational requirements

This comprehensive detection framework ensures that agents can identify and classify errors quickly and accurately, enabling appropriate response strategies and maintaining system reliability in complex, dynamic environments.`
    },
    {
      title: 'Exception Handling Strategies and Graduated Response Mechanisms',
      content: `Exception handling strategies implement sophisticated response mechanisms that adapt to different error types and severities, ensuring appropriate action is taken to maintain system functionality while minimizing disruption and maximizing recovery potential.

**Graduated Response Strategy Framework**

**Immediate Response Level (0-1 seconds)**
The first line of defense involves rapid, automated responses to detected errors:

**Error Logging and Documentation**
- **Structured Logging**: Comprehensive error documentation with standardized format including timestamp, error type, severity, context, and stack traces
- **Context Preservation**: Capture of all relevant system state, input parameters, and environmental conditions at the time of error
- **Error Correlation**: Linking related errors and identifying potential cascading failure patterns
- **Performance Impact Tracking**: Monitoring the computational cost and time impact of error handling processes

**Circuit Breaker Activation**
- **Failure Threshold Detection**: Automatic activation when error rates exceed predefined thresholds for specific components or services
- **State Management**: Tracking circuit breaker states (closed, open, half-open) with appropriate transition logic
- **Selective Blocking**: Preventing requests to failing components while maintaining functionality for healthy services
- **Recovery Monitoring**: Gradual re-enabling of blocked components when health indicators improve

**Short-Term Response Level (1-10 seconds)**

**Retry Mechanisms with Intelligent Backoff**
Sophisticated retry strategies that adapt to error types and historical patterns:

**Exponential Backoff Implementation**
\`\`\`python
async def intelligent_retry(operation, max_attempts=5, base_delay=1.0):
    for attempt in range(max_attempts):
        try:
            result = await operation()
            return result
        except Exception as e:
            if attempt == max_attempts - 1:
                raise e
            
            # Calculate delay with jitter to prevent thundering herd
            delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
            
            # Adapt delay based on error type
            if isinstance(e, RateLimitError):
                delay *= 2  # Longer delays for rate limiting
            elif isinstance(e, NetworkError):
                delay *= 1.5  # Moderate delays for network issues
            
            await asyncio.sleep(delay)
\`\`\`

**Retry Strategy Selection**
- **Error-Type Specific Retries**: Different retry patterns for network errors, rate limits, and transient failures
- **Progressive Backoff**: Increasing delays between retry attempts to reduce system load
- **Jitter Implementation**: Random delay variations to prevent synchronized retry storms
- **Success Rate Monitoring**: Dynamic adjustment of retry parameters based on historical success rates

**Fallback Mechanism Activation**
When primary operations fail, sophisticated fallback systems engage:

**Tool and Service Fallbacks**
- **Alternative Tool Selection**: Automatic switching to backup tools or services when primary options fail
- **Degraded Functionality**: Provision of simplified or approximate results when full functionality is unavailable
- **Cached Response Utilization**: Serving previously cached results when real-time data is unavailable
- **Simplified Workflow Paths**: Alternative processing routes that bypass failing components

**Medium-Term Response Level (10 seconds - 5 minutes)**

**Graceful Degradation Implementation**
Sophisticated systems for maintaining partial functionality during extended failures:

**Capability Assessment and Adaptation**
- **Functionality Mapping**: Dynamic assessment of which capabilities remain available during various failure scenarios
- **User Communication**: Clear, informative messages about reduced functionality and expected resolution timeframes
- **Alternative Workflow Provision**: Guidance for users to accomplish goals through available alternative methods
- **Resource Reallocation**: Dynamic redistribution of resources to maintain critical functionality

**Self-Correction and Replanning**
Advanced agents implement autonomous correction mechanisms:
- **Error Root Cause Analysis**: Automated analysis of error patterns to identify underlying causes
- **Strategy Adjustment**: Modification of agent behavior, parameters, or approaches based on failure analysis
- **Plan Regeneration**: Creation of alternative execution plans that avoid identified failure points
- **Learning Integration**: Incorporation of error insights into future decision-making processes

**Long-Term Response Level (5+ minutes)**

**System Recovery and Restoration**
Comprehensive recovery mechanisms for extended failures:

**State Rollback and Recovery**
- **Checkpoint Systems**: Regular saving of system state to enable rollback to stable configurations
- **Transaction Management**: Atomic operations that can be safely rolled back when failures occur
- **Data Integrity Verification**: Comprehensive checks to ensure system consistency after recovery operations
- **Progressive Restoration**: Gradual re-enabling of functionality with validation at each step

**Escalation Protocols**
Structured escalation for issues requiring human intervention:
- **Severity-Based Escalation**: Automatic escalation based on error impact and system criticality
- **Time-Based Escalation**: Progressive escalation when automated recovery attempts exceed time thresholds
- **Pattern-Based Escalation**: Escalation triggered by recognition of complex failure patterns requiring human analysis
- **Context-Rich Alerts**: Comprehensive information packages for human operators including error history, attempted recoveries, and system state

**Advanced Handling Patterns**

**Multi-Agent Error Coordination**
Sophisticated error handling in multi-agent systems:

**Distributed Error Response**
- **Error Propagation Patterns**: Controlled sharing of error information across agent networks
- **Coordinated Recovery**: Synchronized recovery efforts across multiple agents
- **Resource Reallocation**: Dynamic redistribution of tasks from failing agents to healthy alternatives
- **Collective Intelligence**: Leveraging insights from multiple agents to improve error resolution

**Adaptive Response Learning**
Machine learning integration for improving error handling over time:
- **Response Effectiveness Tracking**: Monitoring the success rates of different handling strategies
- **Strategy Optimization**: Continuous refinement of response approaches based on outcomes
- **Pattern Recognition**: Identification of new error patterns and development of appropriate responses
- **Predictive Response**: Proactive error handling based on predicted failure scenarios

**User-Centric Error Handling**
Error handling strategies that prioritize user experience:

**Transparent Communication**
- **Clear Error Messages**: User-friendly explanations of what went wrong and what is being done to fix it
- **Progress Updates**: Regular communication about recovery efforts and expected resolution times
- **Alternative Options**: Immediate provision of workarounds or alternative approaches
- **Feedback Mechanisms**: Channels for users to report additional context or request specific assistance

**Context-Sensitive Responses**
- **User Role Adaptation**: Different error handling approaches based on user expertise and role
- **Task Criticality Assessment**: Prioritizing recovery efforts based on the importance of the affected functionality
- **Personalized Recovery Options**: Tailored recovery suggestions based on user history and preferences
- **Graceful Degradation Paths**: Carefully designed fallback experiences that maintain user productivity

This comprehensive approach to exception handling ensures that agents can respond appropriately to various failure scenarios while maintaining user trust and system reliability through sophisticated, adaptive response mechanisms.`
    },
    {
      title: 'Recovery Mechanisms and State Management for System Resilience',
      content: `Recovery mechanisms represent the sophisticated systems that restore agents to stable, operational states after errors occur, encompassing state management, self-correction capabilities, and intelligent restoration strategies that ensure long-term system resilience and reliability.

**Comprehensive State Management Architecture**

**Multi-Level State Preservation**
Robust recovery requires sophisticated state management across multiple system layers:

**Application State Management**
- **Agent Memory State**: Preservation of conversation history, learned patterns, and contextual information
- **Goal State Tracking**: Maintenance of current objectives, progress indicators, and milestone achievements
- **Configuration State**: Backup of agent settings, preferences, and operational parameters
- **Workflow State**: Checkpointing of multi-step processes and task execution progress

**System State Management**
- **Resource State**: Tracking of allocated resources, open connections, and active processes
- **Service State**: Monitoring of external service connections, authentication tokens, and API quotas
- **Performance State**: Historical metrics, baseline measurements, and optimization parameters
- **Security State**: Authentication status, permission levels, and security context preservation

**Data State Management**
- **Transaction State**: Atomic operation tracking with rollback capabilities
- **Cache State**: Preservation of cached data with validity tracking and refresh mechanisms
- **Version State**: Multiple state versions for rollback and comparison purposes
- **Consistency State**: Data integrity checks and cross-reference validation

**Intelligent Recovery Strategies**

**Automated Self-Correction Mechanisms**
Advanced agents implement sophisticated self-correction capabilities:

**Error Pattern Analysis and Response**
\`\`\`python
class SelfCorrectionEngine:
    def __init__(self):
        self.error_patterns = {}
        self.correction_strategies = {}
        self.success_metrics = {}
    
    async def analyze_and_correct(self, error_context, system_state):
        # Analyze error pattern
        pattern = self.identify_error_pattern(error_context)
        
        # Select appropriate correction strategy
        strategy = self.select_correction_strategy(pattern, system_state)
        
        # Apply correction with monitoring
        correction_result = await self.apply_correction(strategy, error_context)
        
        # Validate correction effectiveness
        if self.validate_correction(correction_result):
            self.update_success_metrics(strategy, True)
            return correction_result
        else:
            self.update_success_metrics(strategy, False)
            return await self.escalate_for_manual_correction(error_context)
    
    def identify_error_pattern(self, error_context):
        # Pattern recognition based on error characteristics
        pattern_key = f"{error_context.error_type}_{error_context.agent_id}"
        
        if pattern_key not in self.error_patterns:
            self.error_patterns[pattern_key] = {
                'frequency': 0,
                'contexts': [],
                'successful_corrections': [],
                'failed_corrections': []
            }
        
        self.error_patterns[pattern_key]['frequency'] += 1
        self.error_patterns[pattern_key]['contexts'].append(error_context)
        
        return pattern_key
    
    def select_correction_strategy(self, pattern, system_state):
        # Strategy selection based on pattern history and current state
        if pattern in self.correction_strategies:
            # Use historical success data to select best strategy
            strategies = self.correction_strategies[pattern]
            return max(strategies, key=lambda s: s.success_rate)
        else:
            # Default strategy for new patterns
            return self.get_default_correction_strategy()
\`\`\`

**Parameter Optimization and Adjustment**
- **Dynamic Parameter Tuning**: Automatic adjustment of agent parameters based on error feedback
- **Threshold Adaptation**: Modification of decision thresholds to prevent recurring errors
- **Strategy Refinement**: Improvement of decision-making strategies based on failure analysis
- **Learning Rate Adjustment**: Optimization of learning parameters to improve future performance

**Proactive Correction Mechanisms**
- **Predictive Error Prevention**: Identification of conditions likely to lead to errors and preemptive action
- **Preventive Strategy Application**: Implementation of strategies to avoid known error scenarios
- **Environmental Adaptation**: Adjustment of behavior based on changing operational conditions
- **Preemptive Resource Management**: Allocation and management of resources to prevent shortage-related errors

**Advanced Recovery Patterns**

**Multi-Agent Recovery Coordination**
Sophisticated recovery in distributed agent systems:

**Distributed Recovery Orchestration**
- **Recovery Task Distribution**: Coordination of recovery efforts across multiple agents
- **Resource Sharing for Recovery**: Temporary allocation of resources from healthy agents to support recovery
- **Parallel Recovery Execution**: Simultaneous recovery operations across different system components
- **Consensus-Based Recovery Decisions**: Collaborative decision-making for complex recovery scenarios

**Cascading Failure Prevention**
- **Failure Isolation**: Prevention of error propagation across agent networks
- **Load Redistribution**: Dynamic reallocation of work from failing agents to healthy alternatives
- **Graceful Service Degradation**: Coordinated reduction of functionality across the system
- **Recovery Priority Management**: Intelligent prioritization of recovery efforts based on system criticality

**State Restoration Techniques**

**Incremental Recovery Approaches**
Rather than complete system resets, sophisticated systems implement incremental recovery:

**Selective State Rollback**
- **Component-Level Rollback**: Rolling back only affected system components while preserving healthy state
- **Temporal State Selection**: Choosing optimal rollback points based on error timing and impact
- **Dependency-Aware Rollback**: Consideration of component dependencies during rollback operations
- **Validation-Driven Restoration**: Comprehensive validation of restored state before resuming operations

**Progressive Recovery Validation**
- **Health Check Implementation**: Systematic verification of system health after recovery operations
- **Functionality Testing**: Automated testing of recovered components before full restoration
- **Performance Validation**: Verification that recovered systems meet performance requirements
- **Integration Testing**: Validation of interactions between recovered and unchanged components

**Learning-Enhanced Recovery**

**Recovery Experience Integration**
Advanced systems learn from recovery experiences to improve future performance:

**Recovery Pattern Learning**
- **Success Pattern Recognition**: Identification of recovery strategies that consistently produce positive outcomes
- **Failure Mode Analysis**: Analysis of recovery attempts that failed to identify improvement opportunities
- **Context-Strategy Matching**: Learning optimal recovery strategies for different error contexts
- **Timing Optimization**: Learning optimal timing for recovery attempts and resource allocation

**Adaptive Recovery Strategy Evolution**
- **Strategy Effectiveness Tracking**: Continuous monitoring of recovery strategy success rates
- **Dynamic Strategy Generation**: Creation of new recovery strategies based on learned patterns
- **Strategy Combination Optimization**: Learning to combine multiple recovery approaches for maximum effectiveness
- **Environmental Adaptation**: Adjustment of recovery strategies based on changing operational environments

**Recovery Performance Optimization**

**Efficiency-Focused Recovery Design**
Recovery systems balance thoroughness with efficiency:

**Resource-Aware Recovery**
- **Computational Budget Management**: Allocation of recovery resources based on error severity and system capacity
- **Time-Bounded Recovery**: Implementation of recovery time limits to prevent indefinite recovery attempts
- **Priority-Based Resource Allocation**: Dynamic resource allocation based on recovery task criticality
- **Parallel Recovery Processing**: Simultaneous execution of multiple recovery operations when possible

**User Impact Minimization**
- **Background Recovery Operations**: Execution of recovery processes without interrupting user interactions
- **Progressive Service Restoration**: Gradual restoration of functionality to minimize user disruption
- **Transparent Recovery Communication**: Clear communication about recovery progress without overwhelming users
- **Alternative Service Provision**: Provision of alternative functionality during recovery operations

This comprehensive recovery framework ensures that agents can not only detect and handle errors effectively but also learn from failure experiences to build increasingly robust and resilient systems that maintain high availability and user satisfaction even in challenging operational environments.`
    },
    {
      title: 'ADK Implementation Patterns and Multi-Agent Error Handling',
      content: `Google's Agent Development Kit (ADK) provides sophisticated patterns for implementing robust exception handling across multi-agent systems, offering architectural approaches that distribute error handling responsibilities while maintaining coordinated recovery capabilities.

**ADK SequentialAgent Architecture for Error Handling**

**Layered Agent Responsibility Model**
The ADK SequentialAgent pattern enables sophisticated error handling through specialized agent roles:

**Primary Handler Agent**
Focused on core functionality with error detection:
\`\`\`python
from google.adk.agents import Agent, SequentialAgent

primary_handler = Agent(
    name="primary_operation_agent",
    model="gemini-2.0-flash-exp",
    instruction="""
You are the primary operation handler responsible for executing the main task.

Your responsibilities:
1. Execute the primary operation using appropriate tools
2. Validate results and detect any errors or anomalies
3. Set state flags to indicate success or failure:
   - state["operation_successful"] = True/False
   - state["error_details"] = error information if applicable
   - state["partial_results"] = any useful data even if operation failed
4. Provide clear status reporting for downstream agents

Error Detection Guidelines:
- Check for tool execution failures or exceptions
- Validate output format and completeness
- Assess result quality against expected standards
- Document any issues for error handling agents
    """,
    tools=[primary_operation_tool]
)
\`\`\`

**Error Detection and Analysis Agent**
Specialized agent for comprehensive error analysis:
\`\`\`python
error_analysis_agent = Agent(
    name="error_analysis_specialist",
    model="gemini-2.0-flash-exp",
    instruction="""
You are an error analysis specialist responsible for comprehensive error assessment.

Your responsibilities:
1. Analyze state["operation_successful"] and state["error_details"]
2. Classify error type and severity:
   - Transient errors that may resolve with retry
   - Configuration errors requiring parameter adjustment
   - Service availability errors needing fallback strategies
   - Data format errors requiring cleaning or validation
3. Set recovery strategy flags:
   - state["retry_recommended"] = True/False
   - state["fallback_required"] = True/False
   - state["escalation_needed"] = True/False
4. Prepare recovery context for subsequent agents

Analysis Framework:
- Categorize errors by type and impact
- Assess probability of recovery success
- Consider historical patterns and success rates
- Provide detailed recommendations for recovery approach
    """,
    tools=[error_classification_tool, historical_analysis_tool]
)
\`\`\`

**Recovery Strategy Agent**
Dedicated agent for implementing recovery mechanisms:
\`\`\`python
recovery_strategy_agent = Agent(
    name="recovery_strategy_executor",
    model="gemini-2.0-flash-exp", 
    instruction="""
You are a recovery strategy executor responsible for implementing error recovery.

Your responsibilities:
1. Check recovery flags from error analysis:
   - If state["retry_recommended"], implement retry with backoff
   - If state["fallback_required"], execute fallback procedures
   - If state["escalation_needed"], prepare escalation documentation
2. Execute appropriate recovery actions:
   - Retry primary operation with adjusted parameters
   - Use alternative tools or services for fallback
   - Implement graceful degradation when necessary
3. Update state with recovery results:
   - state["recovery_attempted"] = True
   - state["recovery_successful"] = True/False
   - state["final_result"] = recovered or degraded result

Recovery Implementation:
- Apply exponential backoff for retries
- Use circuit breaker patterns for failing services
- Implement intelligent fallback selection
- Document all recovery attempts and outcomes
    """,
    tools=[retry_tool, fallback_service_tool, degradation_tool]
)
\`\`\`

**Response Synthesis Agent**
Final agent for coordinating user communication:
\`\`\`python
response_synthesis_agent = Agent(
    name="response_coordinator",
    model="gemini-2.0-flash-exp",
    instruction="""
You are responsible for synthesizing final responses with appropriate error communication.

Your responsibilities:
1. Review all state information from previous agents
2. Determine the appropriate response based on:
   - Primary operation success/failure
   - Recovery attempts and results
   - Available partial or fallback data
3. Craft user-appropriate responses:
   - Success: Present results clearly and completely
   - Partial success: Explain limitations and provide available information
   - Failure: Apologize, explain situation, provide alternatives
4. Include appropriate disclaimers and next steps

Response Guidelines:
- Be transparent about limitations without exposing technical details
- Provide constructive alternatives when possible
- Maintain user confidence while being honest about issues
- Include escalation information when appropriate
    """,
    tools=[]
)
\`\`\`

**Complete ADK Error Handling Pipeline**
\`\`\`python
class ADKRobustOperationAgent:
    def __init__(self):
        """Initialize comprehensive ADK-based error handling system."""
        
        # Setup specialized tools for each agent
        self.setup_operation_tools()
        
        # Create the sequential agent pipeline
        self.robust_agent = SequentialAgent(
            name="robust_operation_pipeline",
            sub_agents=[
                self.create_primary_handler(),
                self.create_error_analyzer(),
                self.create_recovery_executor(),
                self.create_response_coordinator()
            ]
        )
        
        # Initialize monitoring and metrics
        self.operation_metrics = {
            "total_operations": 0,
            "primary_success_rate": 0.0,
            "recovery_success_rate": 0.0,
            "escalation_rate": 0.0,
            "average_response_time": 0.0
        }
    
    def create_primary_handler(self):
        """Create primary operation handler with error detection."""
        return Agent(
            name="primary_handler",
            model="gemini-2.0-flash-exp",
            instruction=self.get_primary_handler_instruction(),
            tools=[self.operation_tools["primary_tool"]]
        )
    
    def create_error_analyzer(self):
        """Create specialized error analysis agent."""
        return Agent(
            name="error_analyzer", 
            model="gemini-2.0-flash-exp",
            instruction=self.get_error_analyzer_instruction(),
            tools=[self.operation_tools["analysis_tool"]]
        )
    
    def create_recovery_executor(self):
        """Create recovery strategy execution agent."""
        return Agent(
            name="recovery_executor",
            model="gemini-2.0-flash-exp",
            instruction=self.get_recovery_executor_instruction(),
            tools=[
                self.operation_tools["retry_tool"],
                self.operation_tools["fallback_tool"],
                self.operation_tools["degradation_tool"]
            ]
        )
    
    def create_response_coordinator(self):
        """Create final response coordination agent."""
        return Agent(
            name="response_coordinator",
            model="gemini-2.0-flash-exp", 
            instruction=self.get_response_coordinator_instruction(),
            tools=[]
        )
    
    async def execute_robust_operation(self, operation_request):
        """
        Execute operation through robust error handling pipeline.
        
        Args:
            operation_request: The operation to perform
            
        Returns:
            Comprehensive result with error handling status
        """
        
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Execute the robust agent pipeline
            result = await self.robust_agent.run(
                input_text=operation_request,
                state={
                    "request": operation_request,
                    "start_time": start_time,
                    "operation_id": self.generate_operation_id()
                }
            )
            
            # Analyze final state for metrics
            final_state = result.get("state", {})
            
            # Update operational metrics
            self.update_metrics(final_state, start_time)
            
            return {
                "success": final_state.get("operation_successful", False),
                "recovery_used": final_state.get("recovery_attempted", False),
                "escalated": final_state.get("escalation_needed", False),
                "response": result.get("output", ""),
                "execution_time": asyncio.get_event_loop().time() - start_time,
                "operation_id": final_state.get("operation_id"),
                "pipeline_effective": True
            }
            
        except Exception as e:
            # Handle pipeline-level errors
            return await self.handle_pipeline_failure(e, operation_request, start_time)
\`\`\`

**Multi-Agent Coordination Patterns**

**Distributed Error Handling Architecture**
ADK enables sophisticated distributed error handling across agent networks:

**Error Broadcasting Patterns**
\`\`\`python
class DistributedErrorCoordinator:
    def __init__(self):
        self.agent_network = {}
        self.error_propagation_rules = {}
        self.recovery_coordination = {}
    
    async def coordinate_error_response(self, error_context, affected_agents):
        """Coordinate error response across multiple agents."""
        
        # Notify all affected agents
        notifications = await self.broadcast_error_notification(
            error_context, affected_agents
        )
        
        # Coordinate recovery efforts
        recovery_plan = await self.create_coordinated_recovery_plan(
            error_context, affected_agents
        )
        
        # Execute coordinated recovery
        recovery_results = await self.execute_coordinated_recovery(
            recovery_plan
        )
        
        return {
            "notifications_sent": len(notifications),
            "recovery_plan": recovery_plan,
            "recovery_results": recovery_results,
            "coordination_successful": all(
                result["success"] for result in recovery_results
            )
        }
\`\`\`

**Resource Sharing for Recovery**
- **Agent Resource Pooling**: Temporary allocation of healthy agent resources to support failing agents
- **Load Redistribution**: Dynamic task reallocation from failing agents to healthy alternatives
- **Backup Agent Activation**: Standby agents that activate when primary agents fail
- **Collective Intelligence**: Leveraging insights from multiple agents for better recovery strategies

**State Synchronization Patterns**
- **Distributed State Consistency**: Ensuring consistent state across agent networks during recovery
- **Checkpoint Coordination**: Coordinated checkpointing for rollback capabilities
- **State Conflict Resolution**: Protocols for resolving state conflicts during recovery
- **Progressive State Restoration**: Coordinated restoration of distributed state

**Advanced ADK Error Handling Features**

**Dynamic Agent Composition**
ADK's flexibility enables dynamic error handling adaptation:
- **Conditional Agent Activation**: Agents that only activate under specific error conditions
- **Error-Specific Agent Selection**: Dynamic selection of recovery agents based on error type
- **Adaptive Pipeline Configuration**: Modification of agent pipelines based on error patterns
- **Runtime Agent Injection**: Addition of specialized recovery agents during error conditions

**Integration with External Systems**
- **Monitoring System Integration**: Connection with external monitoring and alerting systems
- **Logging Infrastructure**: Integration with enterprise logging and analysis platforms
- **Incident Management**: Automatic creation of incident tickets for escalated errors
- **Performance Metrics**: Integration with performance monitoring and analytics systems

This ADK-based approach provides a robust, scalable foundation for implementing sophisticated error handling in production agent systems, leveraging the framework's multi-agent orchestration capabilities to create resilient, self-healing systems.`
    }
  ],

  practicalExamples: [
    {
      title: 'Resilient Customer Service Chatbot with Multi-Layer Error Handling',
      description: 'Enterprise chatbot system with comprehensive exception handling covering database failures, API timeouts, and service degradation with graceful user communication',
      example: 'Customer support system handling order inquiries, account issues, and technical support with robust error recovery',
      steps: [
        'Error Detection Layer: Implement monitoring for database connectivity, API response validation, timeout detection, and natural language processing quality assessment',
        'Primary Handler Setup: Create main chatbot agent with tools for customer database queries, order management systems, and knowledge base searches with built-in error detection',
        'Fallback Strategy Implementation: Design alternative response paths including cached FAQ responses, general help information, and human agent escalation protocols',
        'Recovery Mechanism Design: Implement retry logic with exponential backoff for transient failures, circuit breakers for failing services, and graceful degradation to basic FAQ functionality',
        'User Communication Protocol: Develop transparent error communication that informs users of issues without exposing technical details, provides alternative options, and manages expectations',
        'Escalation and Monitoring: Create automated escalation to human agents for complex issues, comprehensive error logging for analysis, and real-time monitoring dashboards'
      ]
    },
    {
      title: 'Robust Financial Trading Agent with Risk Management',
      description: 'Automated trading system with sophisticated exception handling for market data failures, execution errors, and risk management with regulatory compliance',
      steps: [
        'Market Data Validation: Implement comprehensive validation of price feeds, volume data, and market indicators with anomaly detection and data quality scoring',
        'Execution Error Handling: Create sophisticated handling for order rejections, insufficient funds errors, market closure detection, and trading halt scenarios',
        'Risk Management Integration: Implement circuit breakers for portfolio risk thresholds, position size limits, drawdown protection, and regulatory compliance checks',
        'Fallback Data Sources: Design alternative market data providers with automatic failover, cached historical data for comparison, and manual override capabilities',
        'Recovery and Rollback: Implement transaction rollback for failed trades, position reconciliation after errors, and portfolio rebalancing after recovery',
        'Compliance and Reporting: Create comprehensive audit trails for all errors and recoveries, regulatory reporting for system failures, and real-time risk monitoring'
      ]
    },
    {
      title: 'Smart Manufacturing Quality Control with Adaptive Error Handling',
      description: 'Industrial automation system with robust error handling for sensor failures, equipment malfunctions, and production quality issues with safety protocols',
      example: 'Manufacturing line with robotic assembly, quality inspection, and predictive maintenance with comprehensive error management',
      steps: [
        'Sensor and Equipment Monitoring: Implement real-time monitoring of production sensors, robotic equipment status, and quality measurement devices with predictive failure detection',
        'Safety Protocol Integration: Create immediate safety responses for critical errors, emergency stop procedures, and safe mode operations with comprehensive safety validation',
        'Production Continuity Management: Design alternative production paths around failed equipment, quality control adjustments for sensor failures, and batch processing recovery',
        'Predictive Maintenance Integration: Implement error pattern analysis for predictive maintenance, equipment health scoring, and proactive replacement scheduling',
        'Quality Assurance Recovery: Create rework procedures for quality failures, statistical quality control adjustments, and customer notification protocols for defective products',
        'Supply Chain Coordination: Implement coordination with suppliers for material issues, customer communication for delivery impacts, and inventory management during failures'
      ]
    }
  ],

  references: [
    'Code Complete (2nd ed.) by Steve McConnell - Microsoft Press',
    'Towards Fault Tolerance in Multi-Agent Reinforcement Learning - arXiv:2412.00534',
    'Improving Fault Tolerance and Reliability of Heterogeneous Multi-Agent IoT Systems Using Intelligence Transfer - Electronics, 11(17), 2724',
    'Google Agent Development Kit (ADK) Documentation: https://google.github.io/adk-docs/',
    'Patterns of Fault Tolerant Software by Robert Hanmer',
    'Building Microservices: Designing Fine-Grained Systems by Sam Newman',
    'Site Reliability Engineering: How Google Runs Production Systems'
  ],

  navigation: {
    previous: { href: '/chapters/goal-setting-monitoring', title: 'Goal Setting and Monitoring' },
    next: { href: '/chapters/human-in-loop', title: 'Human-in-the-Loop' }
  }
}
