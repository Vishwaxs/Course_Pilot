<!-- :   -->
# 📝 Faculty Consent Template — CoursePilot

> Use this template to obtain written consent from faculty members before
> ingesting their lecture materials into the CoursePilot system.

---

## Faculty Material Usage Consent Form

### Project: CKG-JTT — Campus Knowledge Graph & Just-In-Time Tutor (CoursePilot)
### Institution: Christ University

---

**Faculty Name:** ___________________________

**Department:** ___________________________

**Course Code(s):** ___________________________

**Date:** ___________________________

---

### Purpose

CoursePilot is an educational technology prototype that:
1. Extracts concepts from lecture slides and audio recordings.
2. Builds a searchable knowledge graph of course topics.
3. Provides AI-assisted question answering for students.
4. Helps faculty identify syllabus coverage gaps.

### Data That Will Be Processed

- [ ] Lecture slide PDFs
- [ ] Lecture audio recordings
- [ ] Lecture notes / supplementary materials
- [ ] Past examination question papers

### How Data Will Be Used

1. **Text extraction**: Content will be converted to plain text for indexing.
2. **Concept extraction**: Key topics and relationships will be identified using NLP.
3. **Embedding & search**: Content will be encoded as vectors for semantic search.
4. **Student queries**: Students will receive AI-generated answers sourced from your materials, with full attribution.

### Data Protection Measures

- Materials are stored on secured university servers only.
- Student-facing responses always include source attribution (slide/lecture ID).
- No personally identifiable student data is linked to faculty materials.
- Materials can be removed at any time upon faculty request.
- The system does not share materials outside the university LMS.

### Consent

I, the undersigned faculty member, **consent / do not consent** (circle one) to the use of my course materials as described above.

I understand that:
- I may withdraw consent at any time by notifying the project coordinator.
- Upon withdrawal, my materials will be removed from the system within 48 hours.
- My materials will only be used for the stated educational purposes.
- Attribution will always be provided to students.

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
