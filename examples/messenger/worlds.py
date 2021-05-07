from parlai.core.worlds import World
from parlai.chat_service.services.messenger.worlds import OnboardWorld
from parlai.core.agents import create_agent_from_shared


class ChattyGooseMessengerOnboardWorld(OnboardWorld):
    """
    Example messenger onboarding world for Chatty Goose.
    """

    @staticmethod
    def generate_world(opt, agents):
        return ChattyGooseMessengerOnboardWorld(opt=opt, agent=agents[0])

    def parley(self):
        self.episodeDone = True


class ChattyGooseMessengerTaskWorld(World):
    """
    Example one person world that talks to a provided agent.
    """
    MODEL_KEY = 'chatty_goose'

    def __init__(self, opt, agent, bot):
        self.agent = agent
        self.episodeDone = False
        self.model = bot
        self.first_time = True

    @staticmethod
    def generate_world(opt, agents):
        if opt['models'] is None:
            raise RuntimeError("Model must be specified")
        return ChattyGooseMessengerTaskWorld(
            opt,
            agents[0],
            create_agent_from_shared(
                opt['shared_bot_params'][ChattyGooseMessengerTaskWorld.MODEL_KEY]
            ),
        )

    @staticmethod
    def assign_roles(agents):
        agents[0].disp_id = 'ChattyGooseAgent'

    def parley(self):
        if self.first_time:
            self.agent.observe(
                {
                    'id': '',
                    'text': 'Welcome to the Chatty Goose demo! '
                            'Please type a query. '
                            'Type [DONE] to finish the chat, or [RESET] to reset the dialogue history.',
                }
            )
            self.first_time = False
        a = self.agent.act()
        if a is not None:
            if '[DONE]' in a['text']:
                self.episodeDone = True
            elif '[RESET]' in a['text']:
                self.model.reset()
                self.agent.observe(
                    {"text": "[History Cleared]", "episode_done": False})
            else:
                self.model.observe(a)
                response = self.model.act()
                # Make sure prefix from agent is not displayed
                response['id'] = ''
                self.agent.observe(response)

    def episode_done(self):
        return self.episodeDone

    def shutdown(self):
        self.agent.shutdown()


class ChattyGooseMessengerOverworld(World):
    """
    World to handle moving agents to their proper places.
    """

    def __init__(self, opt, agent):
        self.agent = agent
        self.opt = opt
        self.first_time = True
        self.episodeDone = False

    @staticmethod
    def generate_world(opt, agents):
        return ChattyGooseMessengerOverworld(opt, agents[0])

    @staticmethod
    def assign_roles(agents):
        for a in agents:
            a.disp_id = 'Agent'

    def episode_done(self):
        return self.episodeDone

    def parley(self):
        if self.first_time:
            self.agent.observe(
                {
                    'id': 'Overworld',
                    'text': 'Welcome to the Chatty Goose Messenger overworld! '
                            'Please type "Start" to start, or "Exit" to exit. ',
                    'quick_replies': ['Start', 'Exit'],
                }
            )
            self.first_time = False
        a = self.agent.act()
        if a is not None and a['text'].lower() == 'exit':
            self.episode_done = True
            return 'EXIT'
        if a is not None and a['text'].lower() == 'start':
            self.episodeDone = True
            return 'default'
        elif a is not None:
            self.agent.observe(
                {
                    'id': 'Overworld',
                    'text': 'Invalid option. Please type "Start".',
                    'quick_replies': ['Start'],
                }
            )
