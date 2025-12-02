CREATE TABLE Subscriptions (
    subscription_id SERIAL PRIMARY KEY,
    tier_name VARCHAR(50) NOT NULL,
    price DECIMAL(5,2) NOT NULL
);

CREATE TABLE Users (
    user_id UUID PRIMARY KEY,
    full_name VARCHAR(100),
    age INT,
    city VARCHAR(50),
    country VARCHAR(50),
    subscription_id INT REFERENCES Subscriptions(subscription_id)
);

CREATE TABLE Payments (
    payment_id SERIAL PRIMARY KEY,
    user_id UUID REFERENCES Users(user_id),
    amount DECIMAL(6,2),
    payment_date TIMESTAMP,
    method VARCHAR(20)
);
