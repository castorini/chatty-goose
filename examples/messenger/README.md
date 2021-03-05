# Deploying a Facebook Messenger Chat Agent

This guide is based on ParlAI's [chat service tutorial](https://parl.ai/docs/tutorial_chat_service.html), where we provide the base configuration and classes for deploying our example `ChattyGooseAgent` as a Facebook Messenger chatbot. The following steps deploy the webhook server to Heroku, however it is also possible to deploy it locally by setting the `local: True` parameter under `opt` in `config.yml`.

1. Create a new [Facebook Page](https://www.facebook.com/pages/create) and [Facebook App for Messenger](https://developers.facebook.com/docs/messenger-platform/getting-started/app-setup) to host the Chatty Goose agent. Add your Facebook Page under the Webhooks settings for Messenger, and check the "messages" subscription field.

2. If deploying to Heroku, create a free account and log into the [Heroku CLI](https://devcenter.heroku.com/articles/heroku-cli) on your machine.

3. Run the webhook server and Chatty Goose agent using our provided configuration. This assumes you have the ParlAI Python package installed and are inside the `chatty-goose` root repository folder.

```
python3.7 -m parlai.chat_service.services.messenger.run --config-path examples/messenger/config.yml
```

4. Add the webhook URL outputted from the above command as a callback URL for the Messenger App settings, and set the verify token to `Messenger4ParlAI`. For Heroku, this URL should look like `https://firstname-parlai-messenger-chatbot.herokuapp.com/webhook`.

5. Visiting your page and sending a message should now trigger the agent to respond!
