"""Entry point for the Multi-Agent Conflict Resolution Assistant.

This script constructs the crew defined in ``crew.py`` and feeds a sample
conversation through it, printing the final outputs in a readable form.
"""
from crew import create_crew

def main() -> None:
    # simple sample conversation 
    sample_conversation = (
        "User: I can't believe you changed the report last minute without telling me.\n"
        "Other: You always overreact, it's not a big deal.\n"
        "User: It's a big deal when I've been working on it all day.\n"
        "Other: Maybe if you were more organized, I wouldn't have to.\n"
        "User: That comment is really hurtful. I put a lot of effort into this.\n"
        "Other: Everyone has to deal with stress, don't be so sensitive.\n"
    )

    crew = create_crew()
    
    # kickoff() now returns a CrewOutput object
    results = crew.kickoff(inputs={"conversation": sample_conversation})

    print("\n" + "="*50)
    print("=== FINAL RESOLUTION REPORT ===")
    print("="*50 + "\n")
    
    # Printing the CrewOutput object automatically yields the final formatted raw string
    print(results)

if __name__ == "__main__":
    main()