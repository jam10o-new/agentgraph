use anyhow::Result;
use mistralrs::{Model, RequestBuilder, SamplingParams, TextMessages, TextMessageRole, ChatCompletionResponse};
use std::sync::Arc;

pub async fn summarize_history(
    model: Arc<Model>,
    history: &[(TextMessageRole, String)],
    sampling: SamplingParams,
) -> Result<String> {
    let mut messages = TextMessages::new()
        .add_message(TextMessageRole::System, "You are a helpful assistant that summarizes conversation history. Be concise and focus on key points and decisions.");
    
    let mut history_text = String::new();
    for (role, content) in history {
        history_text.push_str(&format!("{}: {}\n", role, content));
    }

    messages = messages.add_message(TextMessageRole::User, format!("Please summarize the following conversation history:\n\n{}", history_text));

    let request = RequestBuilder::from(messages).set_sampling(sampling);
    
    let response: ChatCompletionResponse = model.send_chat_request(request).await?;

    if let Some(choice) = response.choices.first() {
        Ok(choice.message.content.clone().unwrap_or_default())
    } else {
        Ok("No summary generated".to_string())
    }
}
