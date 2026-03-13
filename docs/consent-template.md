# Faculty Consent Template — CoursePilot

> Use this template to obtain written consent from faculty members before
> processing their lecture materials with CoursePilot.

---

## Faculty Material Usage Consent Form

### Project: CoursePilot — Document Analytics & Visualization Dashboard
### Institution: Christ University

---

**Faculty Name:** ___________________________

**Department:** ___________________________

**Course Code(s):** ___________________________

**Date:** ___________________________

---

### Purpose

CoursePilot is an educational technology prototype that:
1. Extracts text and images from lecture slide PDFs.
2. Performs NLP analysis (tokenization, NER, concept extraction).
3. Generates visualizations: word clouds, frequency charts, similarity heatmaps.
4. Builds a knowledge graph of course concepts.

### Data That Will Be Processed

- [ ] Lecture slide PDFs
- [ ] Lecture notes / supplementary materials

### How Data Will Be Used

1. **Text extraction**: Content will be converted to plain text for analysis.
2. **Concept extraction**: Key topics and relationships will be identified using NLP.
3. **Visualization**: Word clouds, frequency distributions, and concept graphs will be generated.
4. **Image processing**: Embedded images will be extracted and processed for demonstration.

### Data Protection Measures

- Materials are processed locally only.
- No data is sent to external APIs or cloud services.
- Materials can be removed at any time upon faculty request.

### Consent

I, the undersigned faculty member, **consent / do not consent** (circle one) to the use of my course materials as described above.

I understand that:
- I may withdraw consent at any time by notifying the project coordinator.
- Upon withdrawal, my materials will be removed from the system.
- My materials will only be used for the stated educational purposes.

---

**Signature:** ___________________________

**Date:** ___________________________

**Email:** ___________________________

---

### For Project Administrators

- [ ] Consent form received and verified
- [ ] Materials uploaded with faculty ID tag
- [ ] Consent record stored in audit log
- [ ] Faculty notified upon successful ingestion

**Admin name:** ___________________________
**Date processed:** ___________________________

---

## Digital Consent (Alternative)

For digital consent collection, the **Admin tab** in CoursePilot includes a
consent checkbox workflow. In production:

1. Faculty logs in via campus SSO.
2. Reviews the consent text above in-app.
3. Checks the consent box (recorded with timestamp + user ID).
4. Consent record stored in a tamper-evident audit log.

> **TODO[USER_ACTION]:** Adapt this template to your institution's data
> protection policies and legal requirements (UGC / NAAC guidelines).

---

## Student Data Anonymization

If student performance data (e.g., quiz scores, engagement metrics) is
incorporated in future versions:

1. **Remove PII**: Strip names, enrollment numbers, email addresses.
2. **Aggregate**: Report metrics at cohort level, not individual level.
3. **Pseudonymize**: Use hashed identifiers for any per-student analytics.
4. **Retention**: Delete raw data after the academic term; retain only aggregates.
5. **Access control**: Only authorized faculty and admins may view analytics.

> Refer to UGC Data Protection Guidelines and your institution's IRB
> (Institutional Review Board) for approval before collecting real student data.
