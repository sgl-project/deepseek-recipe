use std::collections::{HashMap, HashSet};

use deepseek_recipe_core::messages::ToolCall;
use deepseek_recipe_core::tools::{ToolChoice, ToolDefinition};

use crate::protocol::client_tools::{
    ClientToolChoice, ClientToolDeclaration, convert_client_tools,
};
use crate::protocol::validation::validate_tool_name;
use crate::request::{
    ConversionError, ConversionOptions, ModelVersion, ResponseOptions, WebSearchBehavior,
};

use super::schema::{
    ResponsesFunctionCall, ResponsesFunctionTool, ResponsesInput, ResponsesInputItem,
    ResponsesNamedToolChoice, ResponsesNamespaceTool, ResponsesTool, ResponsesToolChoice,
    ResponsesToolSearchExecution, ResponsesTypedInputItem,
};

/// The namespace and local name of a namespace tool.
#[derive(Clone, PartialEq, Eq, Hash)]
pub(super) struct ResponsesNamespaceToolName {
    namespace: String,
    local_name: String,
}

/// The declared kind of a registered tool definition.
#[derive(PartialEq)]
enum RegisteredToolKind {
    Function(String),
    Custom(String),
    Namespace(ResponsesNamespaceToolName),
}

/// The kind and definition of a registered function, custom, or namespace
/// tool.
struct RegisteredTool {
    kind: RegisteredToolKind,
    description: Option<String>,
    parameters: Option<serde_json::Value>,
    strict: Option<bool>,
}

#[derive(Default)]
struct DormantTools {
    functions: HashMap<String, ResponsesFunctionTool>,
    custom: HashSet<String>,
    namespaces: HashMap<String, Option<String>>,
    members: HashMap<ResponsesNamespaceToolName, ResponsesFunctionTool>,
}

impl DormantTools {
    fn declare(&mut self, tool: ResponsesTool) -> Result<Option<ResponsesTool>, ConversionError> {
        match tool {
            ResponsesTool::Function(function) if function.defer_loading == Some(true) => {
                self.declare_function(function)?;
                Ok(None)
            }
            ResponsesTool::Custom {
                name,
                defer_loading: Some(true),
            } => {
                if self.functions.contains_key(&name) {
                    return Err(deferred_kind_error(&name));
                }
                self.custom.insert(name);
                Ok(None)
            }
            ResponsesTool::Namespace {
                name,
                description,
                tools,
                defer_loading,
            } => {
                let defer_all = defer_loading == Some(true);
                let mut active = Vec::new();
                let mut has_dormant = false;
                for tool in tools {
                    match tool {
                        ResponsesNamespaceTool::Function(function)
                            if defer_all || function.defer_loading == Some(true) =>
                        {
                            self.declare_member(&name, function)?;
                            has_dormant = true;
                        }
                        tool => active.push(tool),
                    }
                }
                if has_dormant {
                    self.namespaces.insert(name.clone(), description.clone());
                }
                Ok((!active.is_empty()).then_some(ResponsesTool::Namespace {
                    name,
                    description,
                    tools: active,
                    defer_loading: None,
                }))
            }
            tool => Ok(Some(tool)),
        }
    }

    fn declare_function(&mut self, function: ResponsesFunctionTool) -> Result<(), ConversionError> {
        if self.custom.contains(&function.name) {
            return Err(deferred_kind_error(&function.name));
        }
        if let Some(existing) = self.functions.get(&function.name)
            && function_identity(existing) != function_identity(&function)
        {
            return Err(ConversionError::bad_request(format!(
                "tools defers '{}' more than once, with different definitions",
                function.name
            )));
        }
        self.functions.insert(function.name.clone(), function);
        Ok(())
    }

    fn declare_member(
        &mut self,
        namespace: &str,
        function: ResponsesFunctionTool,
    ) -> Result<(), ConversionError> {
        let key = ResponsesNamespaceToolName {
            namespace: namespace.to_owned(),
            local_name: function.name.clone(),
        };
        if let Some(existing) = self.members.get(&key)
            && function_identity(existing) != function_identity(&function)
        {
            return Err(ConversionError::bad_request(format!(
                "tools defers '{}.{}' more than once, with different definitions",
                key.namespace, key.local_name
            )));
        }
        self.members.insert(key, function);
        Ok(())
    }

    fn resolve(&self, tool: ResponsesTool) -> Result<ResponsesTool, ConversionError> {
        match tool {
            ResponsesTool::Function(function) => {
                if self.custom.contains(&function.name) {
                    return Err(deferred_kind_error(&function.name));
                }
                Ok(ResponsesTool::Function(
                    self.resolve_function(function, None),
                ))
            }
            ResponsesTool::Custom {
                name,
                defer_loading: _,
            } => {
                if self.functions.contains_key(&name) {
                    return Err(deferred_kind_error(&name));
                }
                Ok(ResponsesTool::Custom {
                    name,
                    defer_loading: None,
                })
            }
            ResponsesTool::Namespace {
                name,
                description,
                tools,
                defer_loading: _,
            } => {
                let description =
                    description.or_else(|| self.namespaces.get(&name).cloned().unwrap_or_default());
                let tools = tools
                    .into_iter()
                    .map(|tool| match tool {
                        ResponsesNamespaceTool::Function(function) => {
                            ResponsesNamespaceTool::Function(
                                self.resolve_function(function, Some(&name)),
                            )
                        }
                        tool => tool,
                    })
                    .collect();
                Ok(ResponsesTool::Namespace {
                    name,
                    description,
                    tools,
                    defer_loading: None,
                })
            }
            tool => Ok(tool),
        }
    }

    fn resolve_function(
        &self,
        mut function: ResponsesFunctionTool,
        namespace: Option<&str>,
    ) -> ResponsesFunctionTool {
        let declared = match namespace {
            Some(namespace) => self.members.get(&ResponsesNamespaceToolName {
                namespace: namespace.to_owned(),
                local_name: function.name.clone(),
            }),
            None => self.functions.get(&function.name),
        };
        if let Some(declared) = declared {
            if function.description.is_none() {
                function.description = declared.description.clone();
            }
            if function.parameters.is_none() {
                function.parameters = declared.parameters.clone();
            }
            if function.strict.is_none() {
                function.strict = declared.strict;
            }
        }
        function.defer_loading = None;
        function
    }
}

fn function_identity(function: &ResponsesFunctionTool) -> serde_json::Value {
    let mut function = function.clone();
    function.defer_loading = None;
    function.strict = Some(function.strict.unwrap_or(false));
    serde_json::to_value(function).expect("a function tool serializes")
}

fn deferred_kind_error(name: &str) -> ConversionError {
    ConversionError::bad_request(format!(
        "deferred tool '{name}' is referenced with different tool kinds"
    ))
}

/// The tool declarations of a request after name validation, with namespace
/// tools flattened into uniquely named functions.
pub(super) struct ResponsesToolSet {
    client_tools: Vec<ClientToolDeclaration>,
    registered_tools: Vec<RegisteredTool>,
    has_tool_search: bool,
    /// Namespace tools in registration order: the function name and its
    /// namespace and local name. The order keeps
    /// [`Self::convert_function_call`] deterministic when duplicate namespaces
    /// register the same local name.
    namespace_tools: Vec<(String, ResponsesNamespaceToolName)>,
    top_level_names: HashSet<String>,
    model_version: ModelVersion,
    pub(super) discovered_tools: HashMap<usize, Vec<ToolDefinition>>,
    pub(super) response_options: ResponseOptions,
}

impl ResponsesToolSet {
    /// Validate tool names, flatten namespaces, and replace `apply_patch` with
    /// its function tool.
    ///
    /// A search result that repeats the definition of a registered function,
    /// `apply_patch`, or namespace tool exactly reuses that registration and
    /// loads no tool.
    ///
    /// # Errors
    ///
    /// Returns an error, in declaration order, for an invalid function or
    /// namespace name, a custom tool inside a namespace, an invalid or repeated
    /// function name inside a namespace, a custom tool other than
    /// `apply_patch`, or a `web_search` tool under [`WebSearchBehavior::Reject`].
    pub(super) fn from_declarations(
        tools: Option<Vec<ResponsesTool>>,
        options: &ConversionOptions,
        model_version: ModelVersion,
        input: Option<&ResponsesInput>,
    ) -> Result<Self, ConversionError> {
        let input_items = match input {
            Some(ResponsesInput::Items(items)) => Some(items),
            _ => None,
        };
        let resolved_prefix = options.responses_resolved_input_prefix;
        if resolved_prefix > input_items.map_or(0, Vec::len) {
            return Err(ConversionError::bad_request(
                "responses_resolved_input_prefix exceeds the Responses input length",
            ));
        }
        let mut dormant = DormantTools::default();
        let initial = tools
            .unwrap_or_default()
            .into_iter()
            .map(|tool| dormant.declare(tool))
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .flatten()
            .collect();
        if !model_version.supports_mid_conversation_system()
            && (!dormant.functions.is_empty()
                || !dormant.custom.is_empty()
                || !dormant.members.is_empty())
        {
            return Err(ConversionError::bad_request(
                "defer_loading requires a model that supports mid-conversation tool declarations",
            ));
        }
        let mut groups = vec![(None, initial)];
        let mut resolved_search_results = HashMap::new();
        if model_version.supports_mid_conversation_system()
            && let Some(items) = input_items
        {
            for (index, item) in items.iter().enumerate() {
                if let ResponsesInputItem::Typed(ResponsesTypedInputItem::ToolSearchOutput(
                    output,
                )) = item
                {
                    let tools = if index < resolved_prefix {
                        output.tools.clone()
                    } else {
                        let tools = output
                            .tools
                            .iter()
                            .cloned()
                            .map(|tool| dormant.resolve(tool))
                            .collect::<Result<Vec<_>, _>>()?;
                        resolved_search_results.insert(index, tools.clone());
                        tools
                    };
                    groups.push((Some(index), tools));
                }
            }
        }
        let mut converted = Self {
            client_tools: Vec::new(),
            registered_tools: Vec::new(),
            has_tool_search: false,
            namespace_tools: Vec::new(),
            discovered_tools: HashMap::new(),
            top_level_names: groups
                .iter()
                .flat_map(|(_, tools)| tools)
                .filter_map(|tool| match tool {
                    ResponsesTool::Function(function) => Some(function.name.clone()),
                    _ => None,
                })
                .collect(),
            model_version,
            response_options: ResponseOptions {
                resolved_search_results,
                ..ResponseOptions::default()
            },
        };
        for (item_index, tools) in groups {
            let start = converted.client_tools.len();
            let discovered = item_index.is_some();
            for (index, tool) in tools.into_iter().enumerate() {
                let path = item_index.map_or_else(
                    || format!("tools[{index}]"),
                    |item_index| format!("input[{item_index}].tools[{index}]"),
                );
                match tool {
                    ResponsesTool::Function(function) => {
                        validate_tool_name(&function.name, &path)?;
                        let kind = RegisteredToolKind::Function(function.name.clone());
                        let declaration = function.into();
                        if converted
                            .registered_repeat(discovered, &kind, &declaration)
                            .is_none()
                        {
                            converted.register(kind, declaration);
                        }
                    }
                    ResponsesTool::Namespace {
                        name,
                        description,
                        tools,
                        defer_loading: _,
                    } => {
                        validate_tool_name(&name, &path)?;
                        let mut inner_names = HashMap::new();
                        for (inner_index, tool) in tools.into_iter().enumerate() {
                            let mut function = match tool {
                                ResponsesNamespaceTool::Function(function) => function,
                                ResponsesNamespaceTool::Custom { name: custom_name } => {
                                    return Err(ConversionError::bad_request(format!(
                                        "Currently custom tools are not allowed inside a namespace. Found custom tool \
                                     '{custom_name}' in {path}.tools[{inner_index}]."
                                    )));
                                }
                            };
                            let inner_path = format!("{path}.tools[{inner_index}]");
                            validate_tool_name(&function.name, &inner_path)?;
                            if model_version == ModelVersion::DeepseekV41Flash
                                && let Some(namespace_description) = description
                                    .as_deref()
                                    .filter(|description| !description.is_empty())
                            {
                                let function_description =
                                    function.description.as_deref().unwrap_or_default();
                                function.description = Some(format!(
                                    "{namespace_description}\n{function_description}"
                                ));
                            }
                            let tool_name = ResponsesNamespaceToolName {
                                namespace: name.clone(),
                                local_name: function.name.clone(),
                            };
                            let kind = RegisteredToolKind::Namespace(tool_name.clone());
                            let mut declaration = ClientToolDeclaration::from(function);
                            let repeat =
                                converted.registered_repeat(discovered, &kind, &declaration);
                            let registration = repeat.unwrap_or(converted.registered_tools.len());
                            // A namespace may list a name again only as an exact repeat of
                            // the registration that the name already has in it.
                            if inner_names
                                .insert(tool_name.local_name.clone(), registration)
                                .is_some_and(|previous| Some(previous) != repeat)
                            {
                                return Err(ConversionError::bad_request(
                                    "Tool names within a namespace must be unique.",
                                ));
                            }
                            if repeat.is_some() {
                                continue;
                            }
                            declaration.name = unique_name(
                                join_namespace(&name, &tool_name.local_name, model_version),
                                |candidate| converted.is_name_taken(candidate),
                            );
                            converted
                                .namespace_tools
                                .push((declaration.name.clone(), tool_name));
                            converted.register(kind, declaration);
                        }
                    }
                    ResponsesTool::Custom { name, .. } => {
                        if name != "apply_patch" {
                            return Err(ConversionError::bad_request(format!(
                                "Unsupported custom tool: '{name}'. Only 'apply_patch' is supported."
                            )));
                        }
                        let kind = RegisteredToolKind::Custom(name.clone());
                        let declaration = apply_patch_tool(model_version);
                        if converted
                            .registered_repeat(discovered, &kind, &declaration)
                            .is_none()
                        {
                            converted.register(kind, declaration);
                        }
                        converted.response_options.custom_tool_names.insert(name);
                    }
                    ResponsesTool::ToolSearch {
                        execution,
                        description,
                        parameters,
                    } => {
                        let client_tool_search =
                            execution == Some(ResponsesToolSearchExecution::Client);
                        converted.response_options.client_tool_search |=
                            item_index.is_none() && client_tool_search;
                        if model_version.supports_mid_conversation_system() && client_tool_search {
                            converted.has_tool_search = true;
                            converted.client_tools.push(ClientToolDeclaration {
                                name: "tool_search".to_string(),
                                description,
                                parameters,
                                strict: None,
                                defer_loading: false,
                            });
                        }
                    }
                    ResponsesTool::WebSearch(_) | ResponsesTool::WebSearch20250826(_) => {
                        match options.responses_web_search {
                            WebSearchBehavior::Drop => {}
                            WebSearchBehavior::Reject => {
                                return Err(ConversionError::bad_request(
                                    "Server tools are not supported",
                                ));
                            }
                        }
                    }
                    ResponsesTool::Unsupported => {}
                }
            }
            if let Some(item_index) = item_index {
                let tools = converted.client_tools[start..]
                    .iter_mut()
                    .map(|tool| {
                        tool.defer_loading = true;
                        ToolDefinition::from(&*tool)
                    })
                    .collect();
                converted.discovered_tools.insert(item_index, tools);
            }
        }
        Ok(converted)
    }

    /// Return the function names of namespace tools mapped to their namespace
    /// and local name.
    pub(super) fn namespace_tool_map(&self) -> HashMap<String, (String, String)> {
        self.namespace_tools
            .iter()
            .map(|(function_name, name)| {
                (
                    function_name.clone(),
                    (name.namespace.clone(), name.local_name.clone()),
                )
            })
            .collect()
    }

    /// Return whether at least one tool definition was converted.
    pub(super) fn has_tools(&self) -> bool {
        !self.client_tools.is_empty()
    }

    /// Return the name of the client `tool_search` definition, which the
    /// protocol rather than the user declares, when the request has one.
    pub(super) fn tool_search_name(&self) -> Option<&'static str> {
        self.has_tool_search.then_some("tool_search")
    }

    /// Convert a historical function call into a tool call, resolving a
    /// namespaced name to its function name.
    ///
    /// A registered namespace tool uses its registered function name. Other
    /// namespaced names use one unique function name per namespace and local
    /// name pair, recorded in `historical_names`.
    pub(super) fn convert_function_call(
        &self,
        call: ResponsesFunctionCall,
        historical_names: &mut HashMap<ResponsesNamespaceToolName, String>,
    ) -> ToolCall {
        let name = match call.namespace {
            Some(namespace) => {
                let key = ResponsesNamespaceToolName {
                    namespace,
                    local_name: call.name,
                };
                let registered = self
                    .namespace_tools
                    .iter()
                    .find(|(_, name)| *name == key)
                    .map(|(function_name, _)| function_name.clone());
                if let Some(name) = registered {
                    name
                } else if let Some(name) = historical_names.get(&key) {
                    name.clone()
                } else {
                    // Each historical namespace/name pair uses one unique name.
                    let name = unique_name(
                        join_namespace(&key.namespace, &key.local_name, self.model_version),
                        |candidate| {
                            self.is_name_taken(candidate)
                                || historical_names.values().any(|name| name == candidate)
                        },
                    );
                    historical_names.insert(key, name.clone());
                    name
                }
            }
            None => call.name,
        };
        ToolCall {
            id: call.call_id,
            name,
            arguments: call.arguments,
        }
    }

    /// Validate tool schemas and resolve the tool choice.
    ///
    /// A choice that names a `web_search` tool is dropped or rejected per
    /// [`ConversionOptions::responses_web_search`]. A `none` choice or an empty
    /// tool set skips the schema checks and returns no tool definitions;
    /// [`convert_client_tools`] resolves the other choices.
    ///
    /// # Errors
    ///
    /// Returns an error for a choice that names a `web_search` tool under
    /// [`WebSearchBehavior::Reject`], then the errors of
    /// [`convert_client_tools`]: duplicate function names after namespace
    /// flattening, an invalid parameter schema, a named choice that names no
    /// tool, or a named or `required` choice in thinking mode. Last, returns
    /// an error for a `function` choice that names a custom tool or a `custom`
    /// choice that names any other tool.
    pub(super) fn convert_tools(
        self,
        choice: Option<ResponsesToolChoice>,
        thinking_mode: bool,
        options: &ConversionOptions,
    ) -> Result<(Vec<ToolDefinition>, ToolChoice), ConversionError> {
        // Dropping the web_search tool also drops a choice naming it.
        let choice = match choice {
            Some(ResponsesToolChoice::Named(
                ResponsesNamedToolChoice::WebSearch | ResponsesNamedToolChoice::WebSearch20250826,
            )) => match options.responses_web_search {
                WebSearchBehavior::Drop => None,
                WebSearchBehavior::Reject => {
                    return Err(ConversionError::bad_request(
                        "Server tools are not supported",
                    ));
                }
            },
            other => other,
        };
        // Disabling tools skips parameter-schema and final-name checks.
        // Definition conversion has already handled tool names and kinds.
        if matches!(choice, Some(ResponsesToolChoice::None)) {
            return Ok((Vec::new(), ToolChoice::None));
        }
        if self.client_tools.is_empty() {
            return Ok((Vec::new(), ToolChoice::Auto));
        }
        let (choice, named_tool) = match choice {
            Some(ResponsesToolChoice::Named(ResponsesNamedToolChoice::Function { name })) => {
                (ClientToolChoice::Named(name.clone()), Some((name, false)))
            }
            Some(ResponsesToolChoice::Named(ResponsesNamedToolChoice::Custom { name })) => {
                (ClientToolChoice::Named(name.clone()), Some((name, true)))
            }
            Some(ResponsesToolChoice::Required) => (ClientToolChoice::Required, None),
            _ => (ClientToolChoice::Auto, None),
        };
        let converted = convert_client_tools(self.client_tools, choice, thinking_mode)?;
        if let Some((name, custom)) = named_tool
            && self.response_options.custom_tool_names.contains(&name) != custom
        {
            let kind = if custom { "custom" } else { "function" };
            return Err(ConversionError::bad_request(format!(
                "Invalid value for 'tool_choice': '{name}' is not a {kind} tool."
            )));
        }
        Ok(converted)
    }

    /// Return the index in `registered_tools` of the definition of `kind` that
    /// `declaration` repeats, when `declaration` comes from a search result
    /// (`discovered`). Parameter schemas compare as JSON values: key order is
    /// ignored, and `true` differs from `1`.
    fn registered_repeat(
        &self,
        discovered: bool,
        kind: &RegisteredToolKind,
        declaration: &ClientToolDeclaration,
    ) -> Option<usize> {
        if !discovered {
            return None;
        }
        self.registered_tools.iter().position(|tool| {
            tool.kind == *kind
                && tool.description == declaration.description
                && tool.parameters == declaration.parameters
                && tool.strict == declaration.strict
        })
    }

    fn register(&mut self, kind: RegisteredToolKind, declaration: ClientToolDeclaration) {
        self.registered_tools.push(RegisteredTool {
            kind,
            description: declaration.description.clone(),
            parameters: declaration.parameters.clone(),
            strict: declaration.strict,
        });
        self.client_tools.push(declaration);
    }

    fn is_name_taken(&self, candidate: &str) -> bool {
        self.top_level_names.contains(candidate)
            || self
                .namespace_tools
                .iter()
                .any(|(function_name, _)| function_name == candidate)
    }
}

impl From<ResponsesFunctionTool> for ClientToolDeclaration {
    fn from(function: ResponsesFunctionTool) -> Self {
        Self {
            name: function.name,
            description: function.description,
            parameters: function.parameters,
            strict: function.strict,
            defer_loading: false,
        }
    }
}

/// Join a namespace and a local tool name into a function name.
fn join_namespace(namespace: &str, name: &str, model_version: ModelVersion) -> String {
    if model_version == ModelVersion::DeepseekV41Flash {
        format!("{namespace}::{name}")
    } else if namespace.ends_with('_') || name.starts_with('_') {
        format!("{namespace}{name}")
    } else {
        format!("{namespace}__{name}")
    }
}

fn unique_name(base: String, is_taken: impl Fn(&str) -> bool) -> String {
    if !is_taken(&base) {
        return base;
    }
    for index in 1.. {
        let candidate = format!("{base}_{index}");
        if !is_taken(&candidate) {
            return candidate;
        }
    }
    unreachable!()
}

/// Return the function tool that replaces the `apply_patch` custom tool.
fn apply_patch_tool(model_version: ModelVersion) -> ClientToolDeclaration {
    let description = match model_version {
        ModelVersion::DeepseekV4Pro => APPLY_PATCH_DESCRIPTION.replacen(
            "envelope:\n\n*** Begin Patch",
            "envelope:\n*** Begin Patch",
            1,
        ),
        ModelVersion::DeepseekV41Flash => APPLY_PATCH_DESCRIPTION.to_string(),
    };
    ClientToolDeclaration {
        name: "apply_patch".to_string(),
        description: Some(description),
        parameters: Some(serde_json::json!({
            "type": "object",
            "additionalProperties": false,
            "properties": {
                "input": {
                    "type": "string",
                    "description": "The entire contents of the apply_patch command (the *** Begin Patch ... *** End Patch envelope)."
                }
            },
            "required": ["input"]
        })),
        strict: None,
        defer_loading: false,
    }
}

const APPLY_PATCH_DESCRIPTION: &str = "Use the `apply_patch` tool to edit files.\nYour patch language is a stripped‑down, file‑oriented diff format designed to be easy \
   to parse and safe to apply. You can think of it as a high‑level envelope:\n\n*** Begin Patch\n[ one or more file sections ]\n*** \
   End Patch\n\nWithin that envelope, you get a sequence of file operations.\nYou MUST include a header to specify the action you \
   are taking.\nEach operation starts with one of three headers:\n\n*** Add File: <path> - create a new file. Every following line \
   is a + line (the initial contents).\n*** Delete File: <path> - remove an existing file. Nothing follows.\n*** Update File: <path> \
   - patch an existing file in place (optionally with a rename).\n\nMay be immediately followed by *** Move to: <new path> if you \
   want to rename the file.\nThen one or more “hunks”, each introduced by @@ (optionally followed by a hunk header).\nWithin a hunk \
   each line starts with:\n\nFor instructions on [context_before] and [context_after]:\n- By default, show 3 lines of code \
   immediately above and 3 lines immediately below each change. If a change is within 3 lines of a previous change, do NOT duplicate \
   the first change’s [context_after] lines in the second change’s [context_before] lines.\n- If 3 lines of context is insufficient \
   to uniquely identify the snippet of code within the file, use the @@ operator to indicate the class or function to which the \
   snippet belongs. For instance, we might have:\n@@ class BaseClass\n[3 lines of pre-context]\n- [old_code]\n+ [new_code]\n[3 lines \
   of post-context]\n\n- If a code block is repeated so many times in a class or function such that even a single `@@` statement and \
   3 lines of context cannot uniquely identify the snippet of code, you can use multiple `@@` statements to jump to the right \
   context. For instance:\n\n@@ class BaseClass\n@@ \t def method():\n[3 lines of pre-context]\n- [old_code]\n+ [new_code]\n[3 lines \
   of post-context]\n\nThe full grammar definition is below:\nPatch := Begin { FileOp } End\nBegin := \"*** Begin Patch\" \
   NEWLINE\nEnd := \"*** End Patch\" NEWLINE\nFileOp := AddFile | DeleteFile | UpdateFile\nAddFile := \"*** Add File: \" path \
   NEWLINE { \"+\" line NEWLINE }\nDeleteFile := \"*** Delete File: \" path NEWLINE\nUpdateFile := \"*** Update File: \" path \
   NEWLINE [ MoveTo ] { Hunk }\nMoveTo := \"*** Move to: \" newPath NEWLINE\nHunk := \"@@\" [ header ] NEWLINE { HunkLine } [ \"*** \
   End of File\" NEWLINE ]\nHunkLine := (\" \" | \"-\" | \"+\") text NEWLINE\n\nA full patch can combine several operations:\n\n*** \
   Begin Patch\n*** Add File: hello.txt\n+Hello world\n*** Update File: src/app.py\n*** Move to: src/main.py\n@@ def \
   greet():\n-print(\"Hi\")\n+print(\"Hello, world!\")\n*** Delete File: obsolete.txt\n*** End Patch\n\nIt is important to \
   remember:\n\n- You must include a header with your intended action (Add/Delete/Update)\n- You must prefix new lines with `+` even \
   when creating a new file\n- File references can only be relative, NEVER ABSOLUTE.\n";
