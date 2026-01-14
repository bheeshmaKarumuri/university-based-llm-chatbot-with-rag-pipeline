with app.app_context():
    chats = Chat.query.all()
    for chat in chats:
        print(chat.user_id, chat.prompt, chat.response)
