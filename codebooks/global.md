# Codebook for Global Codes: Obstetric Clinicians (three-point scale)

## Table of Contents: 

**Global codes for obstetric clinicians**
- Flow with score from 1 to 3: `[Flow, 1-3]`
- Concerns with score from 1 to 3: `[Concerns, 1-3]`
- Attentive with score from 1 to 3: `[Attentive, 1-3]`
- Warmth with score from 1 to 3: `[Warmth, 1-3]`
- Respect with score from 1 to 3: `[Respect, 1-3]`

**Instructor Notes:**
- 1 is the lowest score and 3 is the highest score
- 1 = low band (original codebook 1-2), 2 = average (original 3), 3 = high band (original 4-5)
- Select a score that is reflective of the entire encounter.

<!-- generated from simax_emo configs/rubric_global_3pt.json (rubric-global-3pt-v1-draft.1, sha256 e1a128230555); do not edit by hand -->

---

### Flow (Clinician only) `[Flow, 1-3]`

#### Definition:

- How easily did the conversation flow between the obstetric clinician and patient?
- Should not count silence during an exam against the obstetric clinician’s score
- Clinician controls most of the conversation; interrupts the patient but doesn’t respond to the patient in a way that hurries the conversation along
- It may be hard to give good flow if the clinician is interrupting but doesn’t necessarily knock them on rushed

#### Scoring Guidelines:

- `[Flow: 1]`: Conversation does not flow well: many awkward pauses, several instances of people talking over one another and cross-talk between the clinician and the patient. The conversation is choppy, with some periods of time when the clinician dominates the conversation. (original codebook level 1-2)
- `[Flow: 2]`: The clinician makes attempts to make the conversation flow, but there are still moments of awkwardness. This is an average conversation. (original codebook level 3)
- `[Flow: 3]`: Conversation flowed very well with good interchanges with appropriate silence in between speakers and no moments of awkwardness. Clinician lets patient talk for long passages without interrupting and asks probing questions to keep the patient talking. (original codebook level 4-5)

---

### Concerns (Clinician only) `[Concerns, 1-3]`

#### Definition:

- How well did the obstetric clinician address patients concerns?
- When the clinician responds with empathy, that would indicate a better concerns score
- Using NURSE or “I wish...” statements would indicate a better concerns score
- Name It: “It sounds like you’ve been worried about what’s going on...”
- Understand: “I know this may be hard.”
- Reassurance (in response to an emotion): “You’ve showed a lot of strength in this situation with you mother.”
- Support: “I’m going to be with you through this process.”
- Explore: “Tell me more.” or “Can you tell me what you’re thinking?”
- Concerns can be physical and/or emotional
- Example: The clinician answers a concern with “you’re right, taking that new drug can make you tired. What other questions do you have?”

#### Scoring Guidelines:

- `[Concerns: 1]`: Clinician gives little or no attention to the patient’s concerns and changes the topic, or makes only some attempts to address them while missing one or two clinical concerns. There are little to no attempts to address the patient’s emotional concerns. (original codebook level 1-2)
- `[Concerns: 2]`: Clinician responds to patient’s clinical concerns and does so consistently. They miss a few emotional concerns. (original codebook level 3)
- `[Concerns: 3]`: Clinician responds to and anticipates all patient concerns, and responds to most (if not all) patient concerns with empathy. At most they miss one or two emotional concerns or a few opportunities for empathic statements. (original codebook level 4-5)

**Note**: Instances of the clinician dismissing a concern (rather than simply missing one) will cause them to lose a point.

---

### Attentive (Clinician only) `[Attentive, 1-3]`

#### Definition:

- How engaged was the oncologist?
- Clinician remembers something from previous visit
- Clinician is paying close attention to the patient
- Not typing while the patient is talking
- Example: One clear indicator is hearing the clinicians typing for a long time (more than a minute)
- Example: The clinician is not hearing the patient because they are typing

#### Scoring Guidelines:

- `[Attentive: 1]`: Clinician seems distracted rather than engaged: they may spend much of the encounter typing, interrupt the patient, or say “mmhmm” throughout the encounter in a way that seems like they are not listening to the patient. There are instances in which they miss something the patient said — asking the same question that the patient already answered or making a comment that shows they did not hear the patient. (original codebook level 1-2)
- `[Attentive: 2]`: Clinician seems to be listening and understanding most of the patient’s story, but do not repeat back much of what the patient is saying to show they are attending. (original codebook level 3)
- `[Attentive: 3]`: Clinician seems in tune with patient and giving the patient their full attention: there are no instances where they forget something the patient said or repeat a question, and they repeat most of what the patient is saying back to them for clarity and understanding. They may even bring up things the patient had said in a prior encounter. (original codebook level 4-5)

**Note**: Any instance of forgetting something that the patient just said, will cause the clinician to lose a point.

---

### Warmth (Clinician only) `[Warmth, 1-3]`

#### Definition:

- How warm do you think the clinician was overall?
- Examples:
- Clinician asking them about their well-being/life/non-medical questions
- Clinician attempts to relate to patient
- The clinician may make comments about the patient’s loved one
- Clinician treats the patient like a whole person (sometimes including family)
- Small talk between the clinician and patient has to be about the patient

#### Scoring Guidelines:

- `[Warmth: 1]`: Clinician seems cold and technical, asking questions mainly to gather data and not providing space for the patient’s story. Responses are superficial or do not sound sincere, and there is no small talk about the patient. (original codebook level 1-2)
- `[Warmth: 2]`: Clinician seems to be technical during parts of the encounter but also warm in other parts. Little small talk about the patient. (original codebook level 3)
- `[Warmth: 3]`: Clinician seems to be warm and genuinely concerned for the patient throughout the entire conversation, asks rapport building questions about the patient’s story. Lots of small talk about the patient. (original codebook level 4-5)

**Note**: Clinician talking about their own life can cause them to lose a point unless they are brief and do so in a way that feels they are connecting with the patient. Then in that case, they may gain a point for warmth.

---

### Respect (Clinician only) `[Respect, 1-3]`

#### Definition:

- How much respect did the clinician convey to the patient?
- Clinician gets points for respect if they ask permission (before examining, before charting, etc) and for the patient’s preferences
- The clinician asks “what would you prefer that I call you?”
- Any time the clinician asks for permission
- “Is it ok if I examine you?”

#### Scoring Guidelines:

- `[Respect: 1]`: Clinician did not show much respect: they may talk down to the patient or disregard them, treat the patient like they don’t know what they are doing, or assume that the patient may not understand or be able to make decisions. (original codebook level 1-2)
- `[Respect: 2]`: Clinician conducted a typical encounter with an average display of recognizing the patient’s dignity. There might be a few instances where there was disrespectful language but rare. (original codebook level 3)
- `[Respect: 3]`: Clinician displayed a lot of respect, took the patient seriously, and spoke in a respectful manner throughout the entire encounter. They showed the patient they recognize them as adults of equal intelligence or status and asked the patient to engage and add to the agenda. (original codebook level 4-5)

**Note**: Clinician talking about their own life can cause them to lose a point unless they are brief and do so in a way that feels they are connecting with the patient. Then in that case, they may gain a point for warmth.

---
