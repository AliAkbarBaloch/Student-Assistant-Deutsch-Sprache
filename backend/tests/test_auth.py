def _register(client, email="student@example.com", password="hunter2", name="Test Student"):
    return client.post(
        "/api/auth/register",
        data={"email": email, "password": password, "name": name},
    )


def test_register_creates_user_and_returns_token(client):
    response = _register(client)
    assert response.status_code == 200

    body = response.json()
    assert body["token"]
    assert body["user"]["email"] == "student@example.com"
    assert body["user"]["name"] == "Test Student"


def test_register_rejects_duplicate_email(client):
    _register(client)
    response = _register(client)
    assert response.status_code == 400


def test_login_with_correct_credentials(client):
    _register(client)
    response = client.post(
        "/api/auth/login",
        data={"email": "student@example.com", "password": "hunter2"},
    )
    assert response.status_code == 200
    assert response.json()["token"]


def test_login_rejects_wrong_password(client):
    _register(client)
    response = client.post(
        "/api/auth/login",
        data={"email": "student@example.com", "password": "wrong-password"},
    )
    assert response.status_code == 401


def test_me_requires_authentication(client):
    response = client.get("/api/auth/me")
    assert response.status_code == 401


def test_me_returns_current_user_with_valid_token(client):
    token = _register(client).json()["token"]
    response = client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200
    assert response.json()["email"] == "student@example.com"
