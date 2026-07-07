# Load env variables and create client
from anthropic import Anthropic
from anthropic.types import ToolParam


# Tools and Schemas

from datetime import datetime, timedelta


client = Anthropic()
model = "claude-haiku-4-5-20251001"






# Helper functions
def add_user_message(messages, text):
    user_message = {"role": "user", "content": text}
    messages.append(user_message)


def add_assistant_message(messages, text):
    assistant_message = {"role": "assistant", "content": text}
    messages.append(assistant_message)


def chat(messages, system=None, temperature=1.0, stop_sequences=None):
    params = {
        "model": model,
        "max_tokens": 1000,
        "messages": messages,
        "temperature": temperature,
        "stop_sequences": stop_sequences or [],
    }

    if system:
        params["system"] = system

    message = client.messages.create(**params)
    # content[0] is only a text block when the model returns text; a tool_use
    # (or empty) response would raise on `.text`, so pick the first text block.
    return next((block.text for block in message.content if block.type == "text"), "")


def add_duration_to_datetime(
    datetime_str, duration=0, unit="days", input_format="%Y-%m-%d"
):
    date = datetime.strptime(datetime_str, input_format)

    if unit == "seconds":
        new_date = date + timedelta(seconds=duration)
    elif unit == "minutes":
        new_date = date + timedelta(minutes=duration)
    elif unit == "hours":
        new_date = date + timedelta(hours=duration)
    elif unit == "days":
        new_date = date + timedelta(days=duration)
    elif unit == "weeks":
        new_date = date + timedelta(weeks=duration)
    elif unit == "months":
        # Month/year math requires integers (list indexing, replace()).
        month = date.month + int(duration)
        year = date.year + month // 12
        month = month % 12
        if month == 0:
            month = 12
            year -= 1
        day = min(
            date.day,
            [
                31,
                29 if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0) else 28,
                31,
                30,
                31,
                30,
                31,
                31,
                30,
                31,
                30,
                31,
            ][month - 1],
        )
        new_date = date.replace(year=year, month=month, day=day)
    elif unit == "years":
        year = date.year + int(duration)
        # Clamp Feb 29 -> Feb 28 when the target year isn't a leap year.
        is_leap = year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)
        day = 28 if (date.month == 2 and date.day == 29 and not is_leap) else date.day
        new_date = date.replace(year=year, day=day)
    else:
        raise ValueError(f"Unsupported time unit: {unit}")

    return new_date.strftime("%A, %B %d, %Y %I:%M:%S %p")


def set_reminder(content, timestamp):
    print(f"----\nSetting the following reminder for {timestamp}:\n{content}\n----")



add_duration_to_datetime_schema = ToolParam(  {
    "name": "add_duration_to_datetime",
    "description": "Adds a specified duration to a datetime string and returns the resulting datetime in a detailed format. This tool converts an input datetime string to a Python datetime object, adds the specified duration in the requested unit, and returns a formatted string of the resulting datetime. It handles various time units including seconds, minutes, hours, days, weeks, months, and years, with special handling for month and year calculations to account for varying month lengths and leap years. The output is always returned in a detailed format that includes the day of the week, month name, day, year, and time with AM/PM indicator (e.g., 'Thursday, April 03, 2025 10:30:00 AM').",
    "input_schema": {
        "type": "object",
        "properties": {
            "datetime_str": {
                "type": "string",
                "description": "The input datetime string to which the duration will be added. This should be formatted according to the input_format parameter.",
            },
            "duration": {
                "type": "number",
                "description": "The amount of time to add to the datetime. Can be positive (for future dates) or negative (for past dates). Defaults to 0.",
            },
            "unit": {
                "type": "string",
                "description": "The unit of time for the duration. Must be one of: 'seconds', 'minutes', 'hours', 'days', 'weeks', 'months', or 'years'. Defaults to 'days'.",
            },
            "input_format": {
                "type": "string",
                "description": "The format string for parsing the input datetime_str, using Python's strptime format codes. For example, '%Y-%m-%d' for ISO format dates like '2025-04-03'. Defaults to '%Y-%m-%d'.",
            },
        },
        "required": ["datetime_str"],
    },
} )

set_reminder_schema = ToolParam( {
    "name": "set_reminder",
    "description": "Creates a timed reminder that will notify the user at the specified time with the provided content. This tool schedules a notification to be delivered to the user at the exact timestamp provided. It should be used when a user wants to be reminded about something specific at a future point in time. The reminder system will store the content and timestamp, then trigger a notification through the user's preferred notification channels (mobile alerts, email, etc.) when the specified time arrives. Reminders are persisted even if the application is closed or the device is restarted. Users can rely on this function for important time-sensitive notifications such as meetings, tasks, medication schedules, or any other time-bound activities.",
    "input_schema": {
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "The message text that will be displayed in the reminder notification. This should contain the specific information the user wants to be reminded about, such as 'Take medication', 'Join video call with team', or 'Pay utility bills'.",
            },
            "timestamp": {
                "type": "string",
                "description": "The exact date and time when the reminder should be triggered, formatted as an ISO 8601 timestamp (YYYY-MM-DDTHH:MM:SS) or a Unix timestamp. The system handles all timezone processing internally, ensuring reminders are triggered at the correct time regardless of where the user is located. Users can simply specify the desired time without worrying about timezone configurations.",
            },
        },
        "required": ["content", "timestamp"],
    },
})

batch_tool_schema = ToolParam( {
    "name": "batch_tool",
    "description": "Invoke multiple other tool calls simultaneously",
    "input_schema": {
        "type": "object",
        "properties": {
            "invocations": {
                "type": "array",
                "description": "The tool calls to invoke",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "The name of the tool to invoke",
                        },
                        "arguments": {
                            "type": "string",
                            "description": "The arguments to the tool, encoded as a JSON string",
                        },
                    },
                    "required": ["name", "arguments"],
                },
            }
        },
        "required": ["invocations"],
    },
})



def get_current_datetime(date_format="%Y-%m-%d %H:%M:%S"):
    print("get_current_datetime tool called")
    if not date_format:
        raise ValueError("date_format cannot be empty")
    
    return datetime.now().strftime(date_format)

get_current_datetime_schema = ToolParam(
    {
    "name": "get_current_datetime",
    "description": "Returns the current date and time formatted according to the specified format",
    "input_schema": {
        "type": "object",
        "properties": {
            "date_format": {
                "type": "string",
                "description": "A string specifying the format of the returned datetime. Uses Python's strftime format codes.",
                "default": "%Y-%m-%d %H:%M:%S"
            }
        },
        "required": []
    }
}
)



if __name__ == "__main__":


    #currentDateTime = get_current_datetime()
   # print(f"Current date and time: {currentDateTime}")


    messages = []
    messages.append({
        "role": "user",
        "content": "What is the exact time, formatted as HH:MM:SS?"
    })

    response = client.messages.create(
        model=model,
        max_tokens=1000,
        messages=messages,
        tools=[get_current_datetime_schema],
    )

    # add the assiastant's response to the messages list
    messages.append({
        "role": "assistant",
        "content": response.content
    })

    # get the tool result and add it to the messages list
    result = get_current_datetime(**response.content[0].input)

    # add the tool result to the messages list
    messages.append({
        "role": "user",
        "content": [{
            "type": "tool_result",
            "tool_use_id": response.content[0].id,
            "content": result,
            "is_error": False
        }]
    })

    print(f"Current Messages: {messages}")

    # Now, let's ask the model to format the current date and time in a different format
    response = client.messages.create(
        model=model,
        max_tokens=1000,
        messages=messages,
        tools=[get_current_datetime_schema]
    )

    print(f"Response: {response.content[0].text}")