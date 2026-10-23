"""Protocol-layer tests. No model needed: with no LLM configured every dossier falls back to a
non-executing quarantine proposal, which is enough to exercise digests, replay, 409s and signatures.
Run: python test_local.py   (needs: pip install -r requirements.txt)"""
import os, json, base64
os.environ["DB_PATH"] = ":memory:"
from fastapi.testclient import TestClient
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives import serialization
import app as A

c = TestClient(A.app)
sk = Ed25519PrivateKey.generate()
x = base64.urlsafe_b64encode(sk.public_key().public_bytes(
    serialization.Encoding.Raw, serialization.PublicFormat.Raw)).rstrip(b"=").decode()
dos = [{"dossierId": f"d{i}", "partition": "stable_core", "receivedAt": "2026-01-01T00:00:00Z",
        "mailbox": "ops@example.test", "objective": "triage",
        "sources": [{"sourceId": f"s{i}", "kind": "email", "provenance": "external", "title": "t",
                     "lines": [{"lineId": f"l{i}a", "text": "hello"}]}]} for i in range(3)]
req = {"profile": A.PROFILE, "operation": "propose", "evaluationId": "ev1",
       "receiptVerifier": {"algorithm": "Ed25519", "publicKeyJwk": {"kty": "OKP", "crv": "Ed25519", "x": x}},
       "corpus": {}, "allowedActions": A.ACTIONS, "dossiers": dos}

r1 = c.post("/", json=req); assert r1.status_code == 200, r1.text
r2 = c.post("/", json=req); assert r2.content == r1.content, "replay must be byte-identical"
bad = json.loads(json.dumps(req)); bad["dossiers"][0]["objective"] = "changed"
assert c.post("/", json=bad).status_code == 409
assert c.post("/", json={"operation": "zzz"}).status_code == 400
dup = json.loads(json.dumps(req)); dup["evaluationId"] = "ev2"; dup["dossiers"][1]["dossierId"] = "d0"
assert c.post("/", json=dup).status_code == 422

resp = r1.json()
def receipts(accept=True, tamper=False):
    out = []
    for i, p in enumerate(resp["proposals"]):
        rc = {"dossierId": p["dossierId"], "callId": p["callId"], "action": p["action"], "accepted": accept and i != 1,
              "proposalDigest": A.proposal_digest(p), "receiptId": f"nonce{i}"}
        msg = A.canon({"profile": A.PROFILE, "evaluationId": "ev1", "inputDigest": resp["inputDigest"], "receipt": rc}).encode()
        rc["receiptSignature"] = base64.b64encode(sk.sign(msg)).decode()
        out.append(rc)
    if tamper: out[0]["accepted"] = not out[0]["accepted"]
    return out
com = {"profile": A.PROFILE, "operation": "commit", "evaluationId": "ev1", "inputDigest": resp["inputDigest"]}
assert c.post("/", json={**com, "receipts": receipts(tamper=True)}).status_code == 400, "tampered receipt must be rejected"
assert c.post("/", json={**com, "evaluationId": "nope", "receipts": receipts()}).status_code == 404
ok = c.post("/", json={**com, "receipts": receipts()}); assert ok.status_code == 200, ok.text
statuses = [o["status"] for o in ok.json()["outcomes"]]; assert statuses == ["executed", "rejected", "executed"], statuses
assert c.post("/", json={**com, "receipts": receipts()}).content == ok.content, "commit replay identical"
print("ALL LOCAL TESTS PASSED")