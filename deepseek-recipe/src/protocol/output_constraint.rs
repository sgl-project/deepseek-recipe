//! Selection of the output constraint shared by the protocol adapters.

use deepseek_recipe_core::conversation::ResponseFormat;
use deepseek_recipe_core::tools::{ToolChoice, ToolDefinition};

use crate::request::{ConversionError, OutputConstraint};

/// Select the output constraint of a converted request. Whether the backend
/// applies it is the caller's decision.
///
/// `tools` and `tool_choice` are the converted client tools, and
/// `protocol_tool` names a definition among `tools` that the protocol adds
/// rather than the user, such as the Responses client `tool_search`; the
/// strictness checks skip it. `response_format`
/// is the format for the constraint: a strict `json_schema` carries its
/// normalized schema here, while the conversation keeps the schema as written.
/// `has_response_format` is whether the request specified a response format at
/// all, including `text` in Chat Completions.
///
/// With tools and a tool choice other than `none`, the constraint is a tool
/// call block; otherwise the response format selects it, and `text` selects
/// none. Thinking mode does not change the constraint; the caller decides
/// how the backend applies it in each mode.
///
/// # Errors
///
/// Returns [`ConversionError::BadRequest`] when `tools` is not empty,
/// `tool_choice` is not `none`, and either the tools other than
/// `protocol_tool` mix strict and non-strict definitions or all of them are
/// strict and `has_response_format` is `true`.
pub(crate) fn resolve_output_constraint(
    tools: &[ToolDefinition],
    protocol_tool: Option<&str>,
    tool_choice: ToolChoice,
    response_format: &ResponseFormat,
    has_response_format: bool,
) -> Result<Option<OutputConstraint>, ConversionError> {
    if !tools.is_empty() && tool_choice != ToolChoice::None {
        let user_tools = || {
            tools
                .iter()
                .filter(|tool| Some(tool.name.as_str()) != protocol_tool)
        };
        if user_tools().any(|tool| tool.strict == Some(true)) {
            if user_tools().any(|tool| tool.strict != Some(true)) {
                return Err(ConversionError::bad_request(
                    "Cannot use strict tool with non-strict tool in the same request",
                ));
            }
            if has_response_format {
                return Err(ConversionError::bad_request(
                    "response_format should not be used with tools",
                ));
            }
        }
        return Ok(Some(OutputConstraint::ToolCalls {
            tools: tools.to_vec(),
            required: tool_choice == ToolChoice::Required,
        }));
    }
    Ok(match response_format {
        ResponseFormat::JsonObject => Some(OutputConstraint::Json { schema: None }),
        ResponseFormat::JsonSchema(schema) => Some(OutputConstraint::Json {
            schema: Some(schema.clone()),
        }),
        ResponseFormat::Regex(regex) => Some(OutputConstraint::Regex(regex.clone())),
        _ => None,
    })
}
